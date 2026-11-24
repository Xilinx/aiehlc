###############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
"""Parse the measured side of a board run log.

Line formats (printed by the runtime on the board):
  * ``LAUNCH split ...`` / ``LAUNCH split_detail ...``  -- src/aiegraph/runtime/main_launch_line.c:37-49
    All *_ms / phase values are per launch in ms; ``pmu_hz`` is the A78 PMU clock (NOT the AIE clock).
    split_detail keeps the runtime-only phases (``rt_*``) apart from host glue (``host_*``).
  * ``PROF part=<i> op=... ms=...`` and ``PROF aie_host_split part=<i> ...``  -- src/aiegraph/runtime/site_prof.c:40-46,65
  * ``[PERF] tag= tile=(c,r) rc= active_cyc= vec_instr= stream_stall= lock_stall=``,
    ``[PERFIO] tag= tile=(c,r) pre_io_active= pre_io_lock_stall= io_active= io_lock_stall=``,
    ``[PERFSUM] tag= tiles= active_cyc min= max= mean=``  -- src/mlir/runtime/aie_runtime.c:427-439
    Counter values are AIE core cycles.

parse(path) -> dict with keys
  split          {kload, bdcfg, startio, coreen, wait_io, core_reset, ms_per_launch, pmu_hz, sum_vs_aie_wall}
  ms_per_launch  float (shortcut of split['ms_per_launch'])
  pmu_hz         int (A78 PMU Hz)
  split_detail   {'calls': {...}, 'rt': {bdcfg, startio, wait_io}, 'host': {glue_bdcfg, flush_startio,
                  inval_wait_io}, 'other': {kload_elf_ms, plan_ms, ..., pmc_delta, xtime_s}}
  splits         list of every (split, split_detail) pair seen, in order (the last one is the one above)
  prof           list of {part, op, ms, bf16_expand_ms[, site_ms, aie_ms]}
  prof_host_split list of {part, kind, <name>_ms ...}
  prof_total     dict from the ``PROF total_nonaie_ms`` line (or None)
  perf           list per launch tag: {tag, tiles, active_max, active_min, active_mean, compute_max,
                  lock_stall_max, stream_stall_max, vec_instr_max, rc_bad, per_tile: [...],
                  perfsum: {...} or None, io: [...]}
                  compute = active - lock_stall - stream_stall (per tile).
"""
from __future__ import annotations

import re
from typing import Any, Dict, List

_KV = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)=(-?[0-9]+(?:\.[0-9]+)?)")
_SPLIT = re.compile(r"LAUNCH split (?!_detail)(kload=.*)$")
_DETAIL = re.compile(r"LAUNCH split_detail calls (.*)$")
_PROF_PART = re.compile(r"PROF part=(\d+) op=(\S+) (.*)$")
_PROF_HS = re.compile(r"PROF aie_host_split part=(-?\d+) kind=(\S+)(.*)$")
_PROF_TOT = re.compile(r"PROF total_nonaie_ms=(.*)$")
_PERF = re.compile(r"\[PERF\] tag=(-?\d+) tile=\((\d+),(\d+)\) rc=(-?\d+) active_cyc=(\d+) vec_instr=(\d+) "
                   r"stream_stall=(\d+) lock_stall=(\d+)")
_PERFIO = re.compile(r"\[PERFIO\] tag=(-?\d+) tile=\((\d+),(\d+)\) pre_io_active=(\d+) pre_io_lock_stall=(\d+) "
                     r"io_active=(\d+) io_lock_stall=(\d+)")
_PERFSUM = re.compile(r"\[PERFSUM\] tag=(-?\d+) tiles=(\d+) active_cyc min=(\d+) max=(\d+) mean=(\d+)")

_CALL_KEYS = ("kload", "bdcfg", "startio", "coreen", "wait_io", "core_reset")


def _num(s: str):
    return float(s) if "." in s else int(s)


def _kv(text: str) -> Dict[str, Any]:
    return {k: _num(v) for k, v in _KV.findall(text)}


def _parse_split(text: str) -> Dict[str, Any]:
    d = _kv(text)
    if "pmu_hz" in d:
        d["pmu_hz"] = int(d["pmu_hz"])
    return d


def _parse_detail(text: str) -> Dict[str, Any]:
    kv = _kv(text)
    calls = {k: int(kv.pop(k)) for k in _CALL_KEYS if k in kv}
    rt = {k[3:]: kv.pop(k) for k in list(kv) if k.startswith("rt_") and k != "rt_phase_sum_ms"}
    host = {k[5:]: kv.pop(k) for k in list(kv) if k.startswith("host_")}
    return {"calls": calls, "rt": rt, "host": host, "other": kv}


def _new_perf(tag: int) -> Dict[str, Any]:
    return {"tag": tag, "per_tile": [], "io": [], "perfsum": None}


def _finish_perf(p: Dict[str, Any]) -> Dict[str, Any]:
    tl = p["per_tile"]
    ok = [t for t in tl if t["rc"] == 0]
    p["tiles"] = len(tl)
    p["rc_bad"] = len(tl) - len(ok)
    for key, out in (("active", "active"), ("compute", "compute"), ("lock_stall", "lock_stall"),
                     ("stream_stall", "stream_stall"), ("vec_instr", "vec_instr")):
        vals = [t[key] for t in ok]
        p[out + "_max"] = max(vals) if vals else None
    act = [t["active"] for t in ok]
    p["active_min"] = min(act) if act else None
    p["active_mean"] = (sum(act) // len(act)) if act else None
    return p


# Host-side console messages ("[16:59:44 DEBUG] main: ...") can land in the middle of a board
# 'LAUNCH ...' line; the line's tail then continues on the next console line.
_HOSTMSG = re.compile(r"\[\d{2}:\d{2}:\d{2} [A-Z]+\] .*$")


def lines(path: str):
    """Console lines with 'LAUNCH ...' lines rejoined when a host message was interleaved into them."""
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        pend = None
        for line in f:
            line = line.rstrip("\r\n")
            if pend is not None:
                line, pend = pend + line, None
            m = _HOSTMSG.search(line) if "LAUNCH " in line else None
            if m and m.start() > line.find("LAUNCH "):
                pend = line[:m.start()]
                continue
            yield line
        if pend is not None:
            yield pend


def parse(path: str) -> Dict[str, Any]:
    splits: List[Dict[str, Any]] = []
    prof: List[Dict[str, Any]] = []
    host_split: List[Dict[str, Any]] = []
    prof_total = None
    perf: List[Dict[str, Any]] = []
    cur = None  # current perf block (one per [PERF] tag run)

    for line in lines(path):
        if "LAUNCH split" in line:
            m = _DETAIL.search(line)
            if m:
                if splits and splits[-1]["split_detail"] is None:
                    splits[-1]["split_detail"] = _parse_detail(m.group(1))
                else:
                    splits.append({"split": None, "split_detail": _parse_detail(m.group(1))})
                continue
            m = _SPLIT.search(line)
            if m:
                splits.append({"split": _parse_split(m.group(1)), "split_detail": None})
            continue
        if "PROF " in line:
            m = _PROF_HS.search(line)
            if m:
                d = {"part": int(m.group(1)), "kind": m.group(2)}
                d.update(_kv(m.group(3)))
                host_split.append(d)
                continue
            m = _PROF_PART.search(line)
            if m:
                d = {"part": int(m.group(1)), "op": m.group(2)}
                d.update(_kv(m.group(3)))
                prof.append(d)
                continue
            m = _PROF_TOT.search(line)
            if m:
                prof_total = _kv("total_nonaie_ms=" + m.group(1))
            continue
        if "[PERF" not in line:
            continue
        m = _PERF.search(line)
        if m:
            tag = int(m.group(1))
            if cur is None or cur["tag"] != tag or cur["perfsum"] is not None:
                if cur is not None:
                    perf.append(_finish_perf(cur))
                cur = _new_perf(tag)
            a, ss, ls = int(m.group(5)), int(m.group(7)), int(m.group(8))
            cur["per_tile"].append({"col": int(m.group(2)), "row": int(m.group(3)), "rc": int(m.group(4)),
                                    "active": a, "vec_instr": int(m.group(6)), "stream_stall": ss,
                                    "lock_stall": ls, "compute": a - ls - ss})
            continue
        m = _PERFIO.search(line)
        if m and cur is not None and cur["tag"] == int(m.group(1)):
            cur["io"].append({"col": int(m.group(2)), "row": int(m.group(3)),
                              "pre_io_active": int(m.group(4)), "pre_io_lock_stall": int(m.group(5)),
                              "io_active": int(m.group(6)), "io_lock_stall": int(m.group(7))})
            continue
        m = _PERFSUM.search(line)
        if m and cur is not None and cur["tag"] == int(m.group(1)):
            cur["perfsum"] = {"tiles": int(m.group(2)), "min": int(m.group(3)), "max": int(m.group(4)),
                              "mean": int(m.group(5))}
    if cur is not None:
        perf.append(_finish_perf(cur))

    last = next((s for s in reversed(splits) if s["split"] is not None), None)
    split = last["split"] if last else None
    return {
        "path": str(path),
        "split": split,
        "split_detail": last["split_detail"] if last else None,
        "splits": splits,
        "ms_per_launch": split.get("ms_per_launch") if split else None,
        "pmu_hz": split.get("pmu_hz") if split else None,
        "prof": prof,
        "prof_host_split": host_split,
        "prof_total": prof_total,
        "perf": perf,
    }
