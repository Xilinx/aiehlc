###############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
"""``emod.ai_offload_prototype(layer)``: ``offload_prototype`` writes the layer's
kernel backbone, then ``claude -p`` fills its ``TODO(...)``s until the kernel is
logically correct.

Whatever Claude says, the verdict is ours (``prototype_check``), run after
every round and also available to Claude as one command:

  todo      no ``TODO`` left in ``<entry>.cc`` / ``<entry>_ref.c``
  kemu      the kernel AND host code run on x86 (``kemu.py``): every output byte
            equals TVM's own output (``make TRACE=1`` dumps) or ``<entry>_ref``
  trace     ``aiehlc-convtile trace``: every load/store matches the derived DMA
  check     ``offload_check``: lint, spec vs every call, reference bit-exact vs
            TVM, whole network with it swapped in
  build     (last, once the others pass; ``build=False`` skips it) aiehlc builds
            ``test_<entry>.cc`` into a board ELF

``kemu`` is the gate that judges the KERNEL: calibrate/emulate in ``check`` run
the reference, so an unfilled backbone is READY there. Files are written to
``<workdir>/aiproto/layerNN_prototype/`` (an existing one is moved aside, never
deleted); Claude may edit only that directory. Nothing runs on a board.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from . import _env
from .errors import AiegraphError

__all__ = ["ProtoVerdict", "AIPrototypeResult", "prototype_check", "ai_offload_prototype"]


@dataclass
class ProtoVerdict:
    """The gates of one ``prototype_check``; ``ok`` when every gate run passed."""

    dir: Path
    entry: str
    rows: list = field(default_factory=list)     # (gate, ok, detail)

    @property
    def ok(self) -> bool:
        return bool(self.rows) and all(ok for _g, ok, _d in self.rows)

    def add(self, gate: str, ok: bool, detail: str) -> bool:
        self.rows.append((gate, ok, detail.strip()))
        return ok

    def __repr__(self) -> str:
        lines = [f"prototype_check  {'PASS' if self.ok else 'FAIL'}  {self.dir / (self.entry + '.cc')}"]
        for g, ok, d in self.rows:
            first, *rest = d.splitlines() or [""]
            lines.append(f"  {g:<6} {'ok  ' if ok else 'FAIL'} {first}")
            lines += [f"              {l}" for l in rest[:30]]
        return "\n".join(lines)


@dataclass
class AIPrototypeResult:
    """Result of ``ai_offload_prototype``: the kernel (path-like) and its last verdict."""

    source: Path
    layer: int
    verdict: ProtoVerdict
    rounds: int
    session: str = None
    log: Path = None

    @property
    def ok(self) -> bool:
        return self.verdict.ok

    def __fspath__(self) -> str:
        return str(self.source)

    def __str__(self) -> str:
        return str(self.source)

    def raise_for_status(self) -> "AIPrototypeResult":
        if not self.ok:
            raise AiegraphError("ai_offload_prototype", f"layer{self.layer}: still FAILS prototype_check",
                                hint=f"see {self.log}")
        return self

    def __repr__(self) -> str:
        lines = [f"ai_offload_prototype  {'CORRECT' if self.ok else 'NOT CORRECT'}  layer{self.layer}  "
                 f"after {self.rounds} round(s)", f"  kernel: {self.source}"]
        lines += ["  " + l for l in repr(self.verdict).splitlines()]
        if self.log:
            lines.append(f"  log:    {self.log}" + (f"  (claude --resume {self.session})" if self.session else ""))
        lines.append(f"  next: emod.offload('layer{self.layer}', '{self.source}')" if self.ok else
                     "  next: read the log, finish by hand, or run again with rounds=")
        return "\n".join(lines)


# ── the verdict ──────────────────────────────────────────────────────────

def _todos(d: Path, entry: str) -> list:
    out = []
    for name in (f"{entry}.cc", f"{entry}_ref.c"):
        f = d / name
        if f.is_file():
            out += [f"{name}:{i}: {l.strip()[:120]}" for i, l in enumerate(f.read_text().splitlines(), 1)
                    if re.search(r"\bTODO\s*[(:]", l)]
    return out


def _trace(cc: Path) -> tuple:
    from .prototype import _convtile
    exe = _convtile()
    if exe is None:
        return True, "skipped: no build/aiehlc-convtile"
    try:
        r = subprocess.run([str(exe), "trace", str(cc)], capture_output=True, text=True, timeout=900)
    except subprocess.TimeoutExpired:
        return False, "aiehlc-convtile trace ran past 900 s"
    out = (r.stdout + r.stderr).strip()
    ok = r.returncode == 0 and "OK:" in out
    lines = out.splitlines()
    return ok, (lines[-1] if ok else "\n".join(lines[-25:]))


def _build(cc_dir: Path, entry: str, timeout_s: float = 1800) -> tuple:
    """aiehlc on ``test_<entry>.cc`` in ``<dir>/aiehlc_build`` (aiehlc.sh writes aout/ in its cwd)."""
    test = cc_dir / f"test_{entry}.cc"
    if not test.is_file():
        return False, f"{test.name} missing (proto.write writes it)"
    work = cc_dir / "aiehlc_build"
    work.mkdir(exist_ok=True)
    log = work / "aiehlc.log"
    t0 = time.time()
    cmd = f'source "{_env.repo_root() / "script/aiehlc.sh"}" --aie-version 5 --runtime-source-file "{test}"'
    with open(log, "w") as f:
        try:
            r = subprocess.run(["bash", "-c", cmd], cwd=work, stdout=f, stderr=subprocess.STDOUT, timeout=timeout_s)
        except subprocess.TimeoutExpired:
            return False, f"aiehlc ran past {timeout_s:.0f} s; log {log}"
    elf = work / "aout" / "worklocal" / "build" / "host"
    if r.returncode == 0 and elf.is_file() and elf.stat().st_mtime >= t0:
        return True, f"ELF {elf} ({time.time() - t0:.0f} s)"
    errs = [l for l in log.read_text(errors="replace").splitlines() if re.search(r"error|Error", l)]
    return False, f"aiehlc rc={r.returncode}, no fresh ELF; log {log}\n" + "\n".join(errs[-20:])


def prototype_check(emod, layer, src_dir, *, entry: str = None, build: bool = False,
                    check: bool = True) -> ProtoVerdict:
    """Is the prototype in *src_dir* a logically correct kernel for *layer*? (see module doc)"""
    from .kemu import kemu
    from .prototype import prototype_for
    proto = prototype_for(emod, layer, None, entry)
    d = Path(src_dir).resolve()
    v = ProtoVerdict(d, proto.entry)
    cc = d / f"{proto.entry}.cc"
    if not cc.is_file():
        v.add("files", False, f"{cc} missing")
        return v
    todo = _todos(d, proto.entry)
    v.add("todo", not todo, f"{len(todo)} TODO line(s) left\n" + "\n".join(todo) if todo else "none left")
    k = kemu(proto, d)
    v.add("kemu", k.ok, f"golden {k.golden}: {k.detail}" + ("" if k.ok else f"\n{k.log}"))
    if "__global__" in cc.read_text():
        v.add("trace", *_trace(cc))
    if check:
        rep = emod.offload_check(f"layer{proto.layer}", cc)
        v.add("check", rep.ok, repr(rep).splitlines()[0] if rep.ok else repr(rep))
    if build and v.ok:
        v.add("build", *_build(d, proto.entry))
    return v


# ── the AI loop ──────────────────────────────────────────────────────────

def _stage_dir(dest: Path) -> None:
    """Never delete: an earlier run's directory is moved aside."""
    if dest.exists():
        keep = dest.with_name(f"{dest.name}.prev-{time.strftime('%Y%m%d-%H%M%S')}")
        dest.rename(keep)
        print(f"[aiegraph] kept the previous prototype: {keep}")
    dest.mkdir(parents=True)


def _prompt(emod, proto, d: Path, verdict: ProtoVerdict, check_cmd: str) -> str:
    repo = _env.repo_root()
    e = proto.entry
    example = repo / "example/tileprogram/ccode/model/resnet18/layer1/aiek_layer01.cc"
    ex = f"\n- A filled-in backbone of this form (ResNet-18 layer 1, scalar, bit-exact): {example}" \
        if example.is_file() and e != "aiek_layer01" else ""
    return f"""You are completing an AIE user kernel for layer {proto.layer} of an int8 network.

aiegraph.offload_prototype generated a BACKBONE for it in {d} (your working directory):
  {e}.cc       AIE kernel {e}_kernel (ports, mesh split, block loop: DONE; compute: TODO),
               host {e}_run (buffers, mesh launch: DONE; staging / parameters / unpacking: TODO),
               AIEK_ENTRY + AIEK_SPEC (DONE, proven against TVM)
  {e}_ref.c    x86 reference of the layer on TVM's layouts{' (COMPLETE: it is the spec of the math)' if proto.requant else ' (TODO if not complete)'}
  test_{e}.cc  board test (main()); do not edit
The layer, its arguments in TVM call order, roles and value facts:
{repr(proto)}
TVM's own C for the layer (the exact computation): {emod.workdir / 'layers' / proto.dir}/

Fill every TODO(...) so the kernel is LOGICALLY CORRECT: the AIE kernel computes the layer
bit-exactly, the host stages TVM's tensors into the kernel's layout, gets the other
parameters to the core (e.g. in the weights' spare channels, as
{repo / 'src/aietensorop/conv2dstem/conv2dstem.cc'} does) and unpacks the output, and
{e}_run returns 0. Delete each TODO comment you resolve.

Verify with (exit 0 = done; it is also our verdict after your turn):
  {check_cmd}
It checks: no TODO left; kemu = your kernel AND host code executed on x86 (each core of the
mesh, aie::load_buffer gathering the real block, aie::commit scattering it) with every output
byte compared to TVM's output; aiehlc-convtile trace (blocks match the DMA); offload_check.

Current verdict:
{verdict!r}

Rules:
- Edit ONLY {e}.cc and, if incomplete, {e}_ref.c in {d}.
- Keep the backbone: ports and aie::shape, the mesh partition, the block loop and its
  load_buffer/store_buffer/release/commit, the tile sizes, AIEK_ENTRY and AIEK_SPEC. All
  reads of a block before its release/commit.
- aiehlc source rules ({repo / 'doc/tutorial_userkernel.md'} sections 2, 6-8): helpers the
  kernel body calls must be #define macros; never put the kernel keyword or the triple-angle
  launch token in a comment or string; nothing reaching the launch is declared above the kernel.
- int32 accumulators; the epilogue exactly as the reference / TVM C rounds.
- Correct first: a scalar kernel is fine. Do not weaken any check.
- When it passes, stop and summarize what you filled in. If blocked, explain why.{ex}
"""


def _follow_up(verdict: ProtoVerdict, check_cmd: str) -> str:
    return f"Our own check still FAILS:\n{verdict!r}\nFix it; verify with:\n  {check_cmd}"


def ai_offload_prototype(emod, layer=None, *, region=None, entry: str = None, rounds: int = 3,
                         build: bool = True, model: str = None, claude: str = "claude",
                         timeout_s: float = 2400, out_dir=None, quiet: bool = False) -> AIPrototypeResult:
    from .aiproto import _run_claude
    from .prototype import Prototypes, prototype_for
    proto = prototype_for(emod, layer, region, entry)
    if isinstance(proto, Prototypes) or not hasattr(proto, "entry"):
        raise AiegraphError("ai_offload_prototype", "name exactly one layer, e.g. \"layer4\"")
    exe = shutil.which(claude)
    if exe is None:
        raise AiegraphError("ai_offload_prototype", f"no `{claude}` on PATH",
                            hint="install Claude Code (https://claude.com/claude-code) or pass claude=<path>")
    d = Path(out_dir).resolve() if out_dir else emod.workdir / "aiproto" / f"layer{proto.layer:02d}_prototype"
    _stage_dir(d)
    proto.write(d)
    lay = f"layer{proto.layer}"
    check_cmd = f'{sys.executable} -m aiegraph protocheck {lay} "{d}" --workdir {emod.workdir}'
    verdict = prototype_check(emod, lay, d, entry=proto.entry)
    tools = ["Read", "Grep", "Glob", f"Edit(/{d}/**)", f"Write(/{d}/**)",
             f"Bash({sys.executable} -m aiegraph protocheck:*)"]
    base = [exe, "-p", "--output-format", "stream-json", "--verbose",
            "--allowedTools", ",".join(tools), "--add-dir", str(d)]
    if model:
        base += ["--model", model]
    log_path, session, ran = d / "ai_prototype.log", None, 0
    print(f"[aiegraph] ai_offload_prototype: {lay} backbone in {d}; asking claude to fill it "
          f"(up to {rounds} round(s), each verified by prototype_check)")
    with open(log_path, "a") as log:
        while ran < rounds and not verdict.ok:
            ran += 1
            prompt = _prompt(emod, proto, d, verdict, check_cmd) if session is None \
                else _follow_up(verdict, check_cmd)
            print(f"[aiegraph] round {ran}/{rounds}: claude -p ...")
            rc, res = _run_claude(base + (["--resume", session] if session else []), prompt, d, log,
                                  timeout_s, quiet)
            session = res.get("session_id") or session
            if res.get("result"):
                print(f"  [claude] {res['result'].strip()[:600]}")
            if rc != 0 and not res:
                raise AiegraphError("ai_offload_prototype", f"claude exited {rc} without a result",
                                    hint=f"see {log_path} (logged in? `claude -p hi`)")
            verdict = prototype_check(emod, lay, d, entry=proto.entry)
    if verdict.ok and build:
        print("[aiegraph] logic verified; building the board test with aiehlc ...")
        verdict = prototype_check(emod, lay, d, entry=proto.entry, build=True)
    out = AIPrototypeResult(d / f"{proto.entry}.cc", proto.layer, verdict, ran, session, log_path)
    print(repr(out))
    return out
