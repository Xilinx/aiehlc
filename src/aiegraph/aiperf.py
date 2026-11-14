###############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
"""Why is an offloaded layer slow, and make it faster -- with ``claude -p``.

    report = emod.AI_perf_analysis("layer1")               # facts + cost model + rules + AI reading
    result = emod.AI_kernel_accelerate("layer1", report)   # AI edits a copy; gated; re-predicted

Both follow skill **kernelperfcheck** (``.cursor/skills/kernelperfcheck/SKILL.md``),
which is handed to Claude verbatim. What Claude says is never the verdict:

* the **facts** are read here, deterministically -- the layer's measured ms
  (``emod.profile``), its kernel source (pragmas, partition, TODOs), the
  generated ``host.cc`` (config calls, MMIO vs control-packet ELF load), the
  kernel ELF (text / data / BSS bytes the loader writes), ``kernel.lst`` (vector
  vs scalar ops) and ``aiehlc.log`` (which pragmas aiehlc actually honoured);
* a **cost model** turns them into estimated ms per component with the measured
  constants (372 ns per config-bus write, ~2.9 ns per control-packet word);
* **rules** (the skill's checklist) give findings without any AI;
* ``AI_kernel_accelerate`` edits only a copy of the kernel, and after every round runs
  our own gates -- ``offload_check`` READY and an aiehlc build of the copy --
  then re-reads the facts from the NEW build and re-predicts. A prediction is
  not a measurement: ``board=`` offloads the copy and measures it.
"""

from __future__ import annotations

import json
import re
import shutil
import struct
import sys
from pathlib import Path
from types import SimpleNamespace

from . import _env
from .aiproto import _digest, _run_claude, _stage
from .errors import AiegraphError
from .report import KernelOptResult, PerfReport

__all__ = ["layer_facts", "cost_model", "rule_findings", "perf_analysis", "kernel_accelerate"]

SKILL = ".cursor/skills/kernelperfcheck/SKILL.md"
WRITE_NS = 372.0        # doc/performance/register_write_cost.md
CTRL_NS = 2.9           # per word over control packets
_WORDS = {"__Runtime_dma_bd_config": 8, "__Runtime_dma_bd_config_multidim_ooo": 12,
          "__Runtime_dma_bd_config_multidim": 12, "__Runtime_startio": 2,
          "__Runtime_dma_channel_enable_ooo": 2}
# The pragmas aiehlc acts on (each logs "[aiehlc] Detected #pragma <name>").
_AIEHLC_PRAGMAS = ("aie_debug_level", "aie_trace", "control_plan_op_control_packet",
                   "CONTROL_PLAN_GROUP_REG_WRITE", "control_plan_dedicated_shim",
                   "control_plan_shim_bd_ctrl", "control_packet_mode", "KERNELCONFIGOFFLOAD",
                   "output_tile_major")
_VEC = ("VMAC", "VMUL", "VLDA", "VLDB", "VST", "VMSC", "VNEGMUL")


# ── facts ────────────────────────────────────────────────────────────────

def elf_segments(path) -> dict:
    """PT_LOAD bytes of a 32-bit ELF: what XAie_LoadElfMem writes (text, data, bss)."""
    out = {"text": 0, "data": 0, "bss": 0}
    p = Path(path)
    if not p.is_file():
        return {}
    b = p.read_bytes()
    if b[:4] != b"\x7fELF" or b[4] != 1:
        return {}
    end = "<" if b[5] == 1 else ">"
    phoff, = struct.unpack_from(end + "I", b, 28)
    phentsize, phnum = struct.unpack_from(end + "HH", b, 42)
    for i in range(phnum):
        typ, _off, _va, _pa, filesz, memsz, flags, _al = struct.unpack_from(end + "8I", b, phoff + i * phentsize)
        if typ != 1:                                    # PT_LOAD
            continue
        if flags & 1:                                   # PF_X
            out["text"] += memsz
        else:
            out["data"] += filesz
            out["bss"] += memsz - filesz
    return out


def _source_facts(src: Path) -> dict:
    text = src.read_text(errors="replace") if src.is_file() else ""
    m = re.search(r"\.partition\(\s*\{([^}]*)\}\s*,\s*(\d+)\s*,\s*(\d+)\s*\)", text)
    return {"pragmas": [l.strip() for l in text.splitlines() if l.strip().startswith("#pragma")],
            "partition": m.group(1).replace(" ", "") if m else None,
            "mesh": [int(m.group(2)), int(m.group(3))] if m else None,
            "todo": len(re.findall(r"TODO\(", text)),
            "backbone": "TODO(compute)" in text,
            "ctrl_init": "ctrl_high_throughput_enable" in text or "control_plane_init" in text}


_DECL_PREFIX = re.compile(r'(?:(?:extern(?:\s+"C")?|static|inline|const|struct|unsigned|signed)\s+)*'
                          r"[A-Za-z_]\w*\s*\**")


def _call_sites(text: str) -> dict:
    """``{name: count}`` of __Runtime_* CALLS -- a prototype (``void f(...);``,
    type name before the identifier) is not a call: host.cc declares
    ``__Runtime_teardown_all`` without ever calling it."""
    calls = {}
    for line in text.splitlines():
        for m in re.finditer(r"(__Runtime_[A-Za-z0-9_]+)\(", line):
            prefix = line[:m.start()].strip()
            if prefix and prefix not in ("return", "else", "do") and _DECL_PREFIX.fullmatch(prefix):
                continue
            calls[m.group(1)] = calls.get(m.group(1), 0) + 1
    return calls


def _host_facts(host: Path, routing: Path) -> dict:
    text = host.read_text(errors="replace") if host.is_file() else ""
    calls = _call_sites(text)
    rt = routing.read_text(errors="replace") if routing.is_file() else ""
    return {"calls": calls,
            "ctrl_elf_load": any(k.startswith("__Runtime_load_kernel_group") and k.endswith("_ctrl")
                                 for k in calls),
            "ctrl_calls": sum(v for k, v in calls.items() if "_ctrl" in k),
            "teardown": calls.get("__Runtime_teardown_all", 0) > 0,
            "routing_calls": len(re.findall(r"\bXAie_[A-Za-z0-9_]+\(", rt))}


def _lst_facts(lst: Path) -> dict:
    if not lst.is_file():
        return {}
    text = lst.read_text(errors="replace")
    vec = {op: len(re.findall(rf"\b{op}\b", text)) for op in _VEC}
    return {"vector_ops": vec, "vector_total": sum(vec.values()),
            "scalar_mac": len(re.findall(r"\bMAC\b", text)), "lines": text.count("\n")}


def _log_facts(log: Path) -> dict:
    text = log.read_text(errors="replace") if log and log.is_file() else ""
    m = re.search(r"out_rounds=(\d+)", text)
    return {"detected_pragmas": sorted(set(re.findall(r"Detected #pragma ([A-Za-z_]+)", text))),
            "out_rounds": int(m.group(1)) if m else None,
            "skip_bss": "skip-bss" in text or "SKIP_BSS" in text,
            "ooo_host_loop": "OOO output outerRounds loop" in text}


def _run_facts(emod, idx: int) -> dict:
    """Measured ms of the layer (profile rows) and its EXIT rc, from the last run."""
    rep = emod.last_run
    log = Path(rep.log) if rep and rep.log else None
    if log is None:
        logs = sorted((emod.workdir / "arm_build").glob("run_*.log"), key=lambda f: f.stat().st_mtime)
        log = logs[-1] if logs else None
    out = {"log": str(log) if log else None, "measured_ms": None, "exit_rc": None,
           "inference_ms": None, "profiled": False}
    if not log or not log.is_file():
        return out
    from .board import parse_console
    text = log.read_text(errors="replace")
    got = parse_console(text)
    out["inference_ms"], out["profiled"] = got["inference_ms"], bool(got["profile"])
    m = re.findall(rf"\[aie-offload\] layer {idx:02d}\b.*EXIT rc=(-?\d+)", text)
    out["exit_rc"] = int(m[-1]) if m else None
    if got["profile"]:
        calls = set(emod.layer(idx).calls)
        out["measured_ms"] = round(sum(ms for c, ms, _w, _o in got["profile"] if c in calls), 3)
    return out


def _offloaded_row(emod, idx: int) -> dict:
    part = emod.workdir / "layers" / "partition.json"
    rows = json.loads(part.read_text()).get("layers", []) if part.is_file() else []
    for r in rows:
        if r.get("index") == idx and r.get("target") == "aie":
            return r
    raise AiegraphError("perf", f"layer{idx} runs on the CPU",
                        hint=f'offload it first: emod.offload(("layer{idx}", "kernel.cc"))')


def layer_facts(emod, layer, source=None, aie_dir=None, aiehlc_log=None) -> dict:
    """Everything the analysis needs about one offloaded layer (no AI)."""
    idxs = emod.resolve(layer)
    if len(idxs) != 1:
        raise AiegraphError("perf", f"{layer!r} names {len(idxs)} layers", hint='one layer, e.g. "layer1"')
    idx, repo = idxs[0], _env.repo_root()
    row = _offloaded_row(emod, idx) if source is None or aie_dir is None else {}
    src = Path(source) if source else emod.recorded_path(row["source"])
    adir = Path(aie_dir) if aie_dir else emod.workdir / row["aie_dir"]
    log = Path(aiehlc_log) if aiehlc_log else emod.workdir / "aiek" / str(row.get("user_kernel")) / "aiehlc.log"
    l = emod.layer(idx)
    return {"layer": f"layer{idx}", "index": idx, "op": l.label, "calls": l.calls,
            "source": str(src), "aie_dir": str(adir), "aiehlc_log": str(log),
            "src": _source_facts(src), "host": _host_facts(adir / "host.cc", adir / "routing.cc"),
            "elf": elf_segments(adir / "build" / "kernel"), "lst": _lst_facts(adir / "build" / "kernel.lst"),
            "aiehlc": _log_facts(log), "run": _run_facts(emod, idx)}


# ── model + rules ────────────────────────────────────────────────────────

def cost_model(f: dict) -> dict:
    """Estimated ms per launch component, from the facts and the measured constants.
    Components the facts cannot price (teardown, compute) are ``None``."""
    cores = (f["src"]["mesh"] or [4, 4])[0] * (f["src"]["mesh"] or [4, 4])[1]
    elf, host = f["elf"], f["host"]
    ns = CTRL_NS if host["ctrl_elf_load"] else WRITE_NS
    words = (elf.get("text", 0) + elf.get("data", 0) + (0 if f["aiehlc"]["skip_bss"] else elf.get("bss", 0))) / 4
    elf_ms = cores * words * ns / 1e6
    cfg_words = sum(host["calls"].get(k, 0) * w for k, w in _WORDS.items()) + 2 * host["routing_calls"]
    cfg_ms = cfg_words * (CTRL_NS if host["ctrl_calls"] else WRITE_NS) / 1e6
    known = elf_ms + cfg_ms
    measured = f["run"]["measured_ms"]
    return {"cores": cores, "elf_load_ms": round(elf_ms, 3), "elf_words_per_core": int(words),
            "config_ms": round(cfg_ms, 3), "config_words": int(cfg_words),
            "teardown": host["teardown"], "computing": not f["src"]["backbone"],
            "known_ms": round(known, 3), "measured_ms": measured,
            "unexplained_ms": round(measured - known, 3) if measured is not None else None}


def rule_findings(f: dict, m: dict) -> list:
    """The skill's checklist, evaluated on the facts. [{id, what, evidence, est_ms, fix}]"""
    out, prag = [], " ".join(f["src"]["pragmas"])
    rc = f["run"]["exit_rc"]
    if f["src"]["backbone"] or (rc is not None and rc != 0):
        out.append({"id": "R0", "what": "the kernel computes nothing (backbone / error exit)",
                    "evidence": f"TODO(compute) in source; EXIT rc={rc}", "est_ms": None,
                    "fix": "fill compute/staging first -- its time ADDS to the launch overhead"})
    if not f["host"]["ctrl_elf_load"]:
        out.append({"id": "R1", "what": "kernel ELF + BSS loaded over the config bus (MMIO)",
                    "evidence": f"{m['cores']} cores x {m['elf_words_per_core']} words x 372 ns "
                                f"(bss {f['elf'].get('bss', 0)} B/core)",
                    "est_ms": m["elf_load_ms"],
                    "fix": "#pragma control_plan_op_control_packet (+ {2,6} partition, ctrl init), or --skip-bss"})
    if f["host"]["teardown"] and "DISABLE_PARTITIONTEARDOWN" not in prag:
        out.append({"id": "R2", "what": "partition torn down (memories zeroed) after every launch",
                    "evidence": "__Runtime_teardown_all in host.cc, no AIE_DEBUG_FLAG_DISABLE_PARTITIONTEARDOWN",
                    "est_ms": None,
                    "fix": "#pragma aie_debug_level(2 | AIE_DEBUG_FLAG_DISABLE_PARTITIONTEARDOWN)"})
    if not f["host"]["ctrl_calls"] and m["config_ms"] > 0.2:
        out.append({"id": "R3", "what": "BD / lock / routing config over MMIO",
                    "evidence": f"~{m['config_words']} config words x 372 ns", "est_ms": m["config_ms"],
                    "fix": "CONTROL_PLAN_GROUP_REG_WRITE, control_plan_shim_bd_ctrl, KERNELCONFIGOFFLOAD"})
    lst = f["lst"]
    if lst and lst["vector_total"] == 0 and not f["src"]["backbone"]:
        out.append({"id": "R5", "what": "scalar kernel (no vector ops in kernel.lst)",
                    "evidence": f"0 VMAC/VMUL, {lst['scalar_mac']} scalar MAC", "est_ms": None,
                    "fix": "aie::mmul<8,8,8,int8,int8> + vector requantize (conv2dstem.cc)"})
    if f["aiehlc"]["ooo_host_loop"] and "output_tile_major" not in prag:
        out.append({"id": "R6", "what": "host loop re-programs shim S2MM BDs per slab (also a race)",
                    "evidence": "aiehlc.log: OOO output outerRounds loop", "est_ms": None,
                    "fix": "#pragma output_tile_major (+ host de-tile)"})
    det = set(f["aiehlc"]["detected_pragmas"])
    used = sorted(set(re.findall(r"#pragma\s+([A-Za-z_]+)", prag)) & set(_AIEHLC_PRAGMAS))
    lost = [p for p in used if p not in det]
    if lost and Path(f["aiehlc_log"]).is_file():
        out.append({"id": "R4", "what": f"aiehlc did not honour #pragma {', '.join(lost)}",
                    "evidence": "no '[aiehlc] Detected #pragma' line in aiehlc.log", "est_ms": None,
                    "fix": "rebuild build/aiehlc (skill stalepragmabinary)"})
    return out


# ── AI ───────────────────────────────────────────────────────────────────

def _claude(claude: str) -> str:
    exe = shutil.which(claude)
    if exe is None:
        raise AiegraphError("perf", f"no `{claude}` on PATH",
                            hint="install Claude Code, pass claude=<path>, or ai=False for the rules only")
    return exe


def _json_block(text: str):
    blocks = re.findall(r"```json\s*(\{.*?\})\s*```", text or "", re.S)
    for b in reversed(blocks):
        try:
            return json.loads(b)
        except ValueError:
            continue
    return None


def _analysis_prompt(f: dict, m: dict, rules: list) -> str:
    repo = _env.repo_root()
    return f"""Analyze why one AIE-offloaded layer of an int8 network is slow, and suggest fixes.

Follow this skill exactly (it has the evidence map, the measured cost constants,
the checklist and the report format):
----- {repo / SKILL}
{(repo / SKILL).read_text()}
-----

Facts collected for {f['layer']} ({f['op']}) -- deterministic, trust them:
{json.dumps(f, indent=1)}

Cost model (estimates from those facts; None = cannot be priced from facts):
{json.dumps(m, indent=1)}

Rule-based findings already found:
{json.dumps(rules, indent=1)}

Read the files the facts point at (kernel source, host.cc, kernel.cc, kernel.lst,
aiehlc.log, the board log) and the reference docs the skill cites. Do not edit
anything. Explain the measured time component by component, confirm or correct
the rule findings, and add what they miss. End your reply with exactly one
```json block in the skill's report format.
"""


def perf_analysis(emod, layer, *, ai: bool = True, model: str = None, claude: str = "claude",
                  timeout_s: float = 900, out_dir=None, quiet: bool = False) -> PerfReport:
    f = layer_facts(emod, layer)
    m = cost_model(f)
    rules = rule_findings(f, m)
    dest = Path(out_dir).resolve() if out_dir else emod.workdir / "aiperf" / f["layer"]
    dest.mkdir(parents=True, exist_ok=True)
    rep = PerfReport(f["layer"], f, m, rules, dest=dest)
    if not f["run"]["profiled"]:
        print(f"[aiegraph] AI_perf_analysis: no profiled board run -- the measured time is unknown "
              f"(emod.run(profile=True) first for a measured breakdown)")
    if ai:
        exe = _claude(claude)
        tools = ["Read", "Grep", "Glob"]
        # Read-only, enforced: --allowedTools only pre-approves, a user's settings
        # may allow more (Bash ran in the first real analysis), so deny the rest.
        cmd = [exe, "-p", "--output-format", "stream-json", "--verbose", "--allowedTools", ",".join(tools),
               "--disallowedTools", "Bash,Edit,Write,NotebookEdit",
               "--add-dir", str(_env.repo_root()), "--add-dir", str(emod.workdir)]
        if model:
            cmd += ["--model", model]
        rep.log = dest / "ai_perf.log"
        print(f"[aiegraph] AI_perf_analysis {f['layer']}: asking claude (read-only, skill kernelperfcheck)...")
        with open(rep.log, "a") as log:
            rc, res = _run_claude(cmd, _analysis_prompt(f, m, rules), dest, log, timeout_s, quiet)
        rep.session, rep.text = res.get("session_id"), res.get("result", "")
        rep.ai = _json_block(rep.text)
        if rep.ai is None:
            print(f"[aiegraph] claude gave no parsable json report (rc={rc}); rule findings only -- see {rep.log}")
    rep.save()
    return rep


def _op_prompt(f: dict, report: PerfReport, entry: Path, check_cmd: str) -> str:
    repo = _env.repo_root()
    sugg = json.dumps(report.suggestions, indent=1)
    return f"""Make one AIE-offloaded layer faster by editing its kernel source.

Skill (method, constants, checklist -- follow it):
----- {repo / SKILL}
{(repo / SKILL).read_text()}
-----

Layer {f['layer']} ({f['op']}), measured {f['run']['measured_ms']} ms per call.
Facts: {json.dumps({k: f[k] for k in ('src', 'host', 'elf', 'lst', 'aiehlc')}, indent=1)}

Suggestions from the analysis, ranked:
{sugg}

Your working copy is {entry} (its directory is a flat copy of the original's).
Edit ONLY files in {entry.parent}. Apply the suggestions in order of expected
saving. Rules:
- Never change the layer's numerics, the AIEK_ENTRY / AIEK_SPEC names or the spec.
- Keep it passing (exit 0, READY):  {check_cmd}
- Control-plane pragmas need their prerequisites (skill section 3.4): one extra
  partition column, the ctrl high-throughput init before device init; see
  {repo / 'src/aietensorop/conv2dstem/conv2dstem.cc'} for the working set.
- After you stop, aiegraph rebuilds the kernel with aiehlc and re-measures the
  facts; a build failure comes back to you.
End with a short list of what you changed and, per change, the suggestion id.
"""


def _gate(emod, idx: int, entry: Path, dest: Path, verbose: bool) -> tuple:
    """Our own verdict on the copy: offload_check READY, then an aiehlc build."""
    chk = emod.offload_check(f"layer{idx}", entry, verbose=verbose)
    if not chk.ok:
        return chk, None, "offload_check NOT READY:\n" + repr(chk)
    _env.frontend("")
    from frontend.tvmrelay.aiekernel.build import aiehlc_compile
    name = re.search(r"AIEK_ENTRY\(\s*(\w+)", entry.read_text())
    spec = SimpleNamespace(name=name.group(1) if name else entry.stem, source=entry)
    built = aiehlc_compile(spec, dest / "build", verbose=verbose)
    if not built.get("ok"):
        tail = "\n".join(Path(built["log"]).read_text(errors="replace").splitlines()[-40:]) \
            if built.get("log") and Path(built["log"]).is_file() else built.get("reason", "")
        return chk, built, f"aiehlc build failed: {built.get('reason')}\n{tail}"
    return chk, built, ""


def _fresh_dest(root: Path, idx: int, orig: Path) -> Path:
    """``aiopt/layer<N>_<stem>``; when the layer already runs a copy in there
    (accelerating an accelerated kernel), the next ``..._r<k>`` instead."""
    dest, k = root / f"layer{idx}_{orig.stem}", 1
    while dest == orig.parent or dest in orig.parents:
        k += 1
        dest = root / f"layer{idx}_{orig.stem}_r{k}"
    return dest


def kernel_accelerate(emod, layer, report: PerfReport, *, rounds: int = 2, board: str = None,
                      host: str = None, model: str = None, claude: str = "claude",
                      timeout_s: float = 2400, out_dir=None, quiet: bool = False,
                      verbose: bool = False) -> KernelOptResult:
    if report is None:
        saved = emod.workdir / "aiperf" / f"layer{emod.resolve(layer)[0]}" / "report.json"
        report = PerfReport.load(saved) if saved.is_file() else perf_analysis(emod, layer, quiet=quiet)
    elif not isinstance(report, PerfReport):
        report = PerfReport.load(report)                 # a path to a saved report
    before = report.facts
    idx = before["index"]
    # The kernel the layer runs NOW (partition.json), not the report's record of
    # it: a report outlives offloads, and a stale path must never be staged.
    orig = emod.source_path(f"layer{idx}")
    if Path(before["source"]).resolve() != orig:
        print(f"[aiegraph] the report was made for {before['source']}; layer{idx} now runs {orig} -- "
              f"optimizing that (re-run AI_perf_analysis for facts of the current kernel)")
        before = dict(before, source=str(orig))
    exe = _claude(claude)
    dest = Path(out_dir).resolve() if out_dir else _fresh_dest(emod.workdir / "aiopt", idx, orig)
    entry = _stage(orig, dest, orig.name)
    digest = _digest(orig)
    check_cmd = f'{sys.executable} -m aiegraph check --workdir {emod.workdir} "{entry}@{idx}"'
    tools = ["Read", "Grep", "Glob", f"Edit(/{dest}/**)", f"Write(/{dest}/**)",
             f"Bash({sys.executable} -m aiegraph check:*)"]
    base = [exe, "-p", "--output-format", "stream-json", "--verbose", "--allowedTools", ",".join(tools),
            "--add-dir", str(dest), "--add-dir", str(_env.repo_root()), "--add-dir", str(emod.workdir)]
    if model:
        base += ["--model", model]
    log_path, session, why, chk, built, n, summary = dest / "ai_kernel_accelerate.log", None, "", None, None, 0, ""
    print(f"[aiegraph] AI_kernel_accelerate layer{idx}: claude edits a copy in {dest} "
          f"(up to {rounds} round(s); gates: offload_check + aiehlc build)")
    with open(log_path, "a") as log:
        for n in range(1, rounds + 1):
            prompt = _op_prompt(before, report, entry, check_cmd) if session is None else \
                f"Our gate rejected the kernel:\n{why}\nFix it (same rules)."
            print(f"[aiegraph] round {n}/{rounds}: claude -p ...")
            rc, res = _run_claude(base + (["--resume", session] if session else []), prompt, dest, log,
                                  timeout_s, quiet)
            session, summary = res.get("session_id") or session, res.get("result", "") or summary
            if rc != 0 and not res:
                raise AiegraphError("perf", f"claude exited {rc} without a result", hint=str(log_path))
            chk, built, why = _gate(emod, idx, entry, dest, verbose)
            if not why:
                break
    if _digest(orig) != digest:
        print(f"[aiegraph] WARNING: {orig} changed during the optimization")
    out = KernelOptResult(entry, orig, f"layer{idx}", n, not why, why, summary, report,
                          session=session, log=log_path)
    if not why:
        wl = Path(built["worklocal"])
        after = layer_facts(emod, f"layer{idx}", source=entry, aie_dir=wl, aiehlc_log=built["log"])
        after["run"] = dict(before["run"])               # not run yet: same measurement as before
        out.after, out.model_after = after, cost_model(after)
        if board:
            out.measured_after = _measure(emod, idx, entry, board, host, verbose)
    print(repr(out))
    return out


def _measure(emod, idx: int, entry: Path, board: str, host, verbose: bool):
    """Offload the copy and run on the board with profiling; the layer's ms."""
    rep = emod.offload((f"layer{idx}", entry), verbose=verbose)
    if not rep.ok:
        print(f"[aiegraph] offload of the optimized kernel failed:\n{rep!r}")
        return None
    emod.run(board, host, profile=True)
    return _run_facts(emod, idx)["measured_ms"]
