"""bdgraph input form of ``estimate()`` -- structural features of one launch read from the aiehlc build JSONs.

Source: ``<worklocal>/dfscheduleprovenancemap.json`` (BD chains, start_io) + ``schedule_view.json`` (host_lines),
the same files Amit's ``bdgraph_extract.py`` reads (amit/tasks/2026-11-20_des-json-gap/, schema settled in the
cost channel 2026-11-20).  This module only COUNTS; it does not price DMA bytes yet:

  trip_count   per looped start_io: ``op['bdgraph']['trip_count']`` if given, else ``start_io.loop_range`` "a..b"
               (numeric in every looped design of the 2026-11-20 corpus scan: 36/36), else
               ``repeat_count x module_attrs.k_rounds`` is NOT assumed -- the channel is marked ``rounds_unknown``.
  bd_configs   per launch = host bd_config lines outside loops + inside-loop lines x trip_count (schedule_view).
  run_b        per BD contiguous DDR run after the runtime's eff_dims trim (rt_seq_bd_desc): trailing wrap<=1 dims
               dropped; one dim of stride 4 with wrap*4 == len is a plain addr/len BD, run_b = len.
  skipped      ``bd_id: "runtime"`` placeholders (BD chosen at run time) are skipped and counted.
  locks        BDs whose acquire AND release lock ids are >= 0 (one id alone = no lock).  Not priced until the
               AM020 lock semantics are confirmed.

Bytes moved per start (repeat_count vs chain vs iter dim) are still an open question to the HW team, so the
bdgraph form returns no DMA/DDR/wait term: ``estimate`` lists them under "unpriced".
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

BD_WRITE_PATHS = ("jit_mmio", "aot_window", "uart_degraded")
_RANGE = re.compile(r"^\s*(-?\d+)\s*\.\.\s*(-?\d+)\s*$")


def _load(d: Path, name: str) -> Optional[Dict[str, Any]]:
    p = d / name
    return json.loads(p.read_text()) if p.exists() else None


def run_bytes(bd: Dict[str, Any]) -> int:
    """Contiguous run (bytes) of one BD after the runtime's eff_dims trim."""
    st, wr = list(bd.get("dim_strides") or []), list(bd.get("dim_wraps") or [])
    n = int(bd.get("len", 0))
    if not st or not wr:
        return n
    eff = len(wr)
    while eff > 1 and wr[eff - 1] <= 1:
        eff -= 1
    if eff == 1 and st[0] == 4 and wr[0] * 4 == n:
        return n
    return int(wr[0]) * 4 if st[0] == 4 else 4


def trip(start: Dict[str, Any], override: Optional[int] = None) -> Optional[int]:
    """Loop trip count of one start_io (1 outside a loop; None if it cannot be read)."""
    if not start.get("inside_scf_for"):
        return 1
    if override is not None:
        return int(override)
    m = _RANGE.match(str(start.get("loop_range", "")))
    return int(m.group(2)) - int(m.group(1)) if m else None


def _host_lines(sv: Dict[str, Any]) -> List[Dict[str, Any]]:
    seen: Dict[int, Dict[str, Any]] = {}
    for t in sv.get("tiles", []):
        for c in t.get("dma_channels", []):
            for h in c.get("host_lines", []):
                seen.setdefault(h["line"], h)
    out, depth = [], 0
    for ln in sorted(seen):
        h = seen[ln]
        if h["kind"] == "loopend":
            depth -= 1
        out.append({"line": ln, "kind": h["kind"], "depth": depth})
        if h["kind"] == "loop":
            depth += 1
    return out


def features(src: Any) -> Dict[str, Any]:
    """src = worklocal dir (str/Path) or {'worklocal': dir, 'trip_count': int?, 'bd_write_path': str?}."""
    spec = dict(src) if isinstance(src, dict) else {"worklocal": src}
    d = Path(spec["worklocal"])
    dfs = _load(d, "dfscheduleprovenancemap.json")
    if dfs is None:
        raise FileNotFoundError(f"no dfscheduleprovenancemap.json in {d}")
    sv = _load(d, "schedule_view.json")
    wp = spec.get("bd_write_path")
    if wp is not None and wp not in BD_WRITE_PATHS:
        raise ValueError(f"bd_write_path {wp!r} not in {BD_WRITE_PATHS}")
    ovr = spec.get("trip_count")

    channels, skipped, locked, trips = [], 0, 0, set()
    rounds_unknown = False
    for t in dfs.get("tiles", []):
        for c in t.get("dma_channels", []):
            bds = []
            for b in c.get("bd_chain", []):
                if not isinstance(b.get("bd_id"), int):
                    skipped += 1
                    continue
                a = (b.get("acquire_lock") or [{}])[0].get("id", -1)
                r = (b.get("release_lock") or [{}])[0].get("id", -1)
                locked += int(a >= 0 and r >= 0)
                bds.append({"bd_id": b["bd_id"], "len": int(b.get("len", 0)), "run_b": run_bytes(b),
                            "packet": bool(b.get("enable_packet"))})
            starts = []
            for s in c.get("start_io", []):
                n = trip(s, ovr)
                rounds_unknown |= n is None
                if n is not None and s.get("inside_scf_for"):
                    trips.add(n)
                starts.append({"repeat_count": int(s.get("repeat_count", 1)), "trip": n})
            channels.append({"tile": (t.get("col"), t.get("row")), "type": t.get("type"),
                             "dir": c.get("direction"), "ch": c.get("channel"), "bds": bds, "starts": starts})

    per = {"launch": 0, "round": 0}
    waits = {"launch": 0, "round": 0}
    if sv is not None:
        for h in _host_lines(sv):
            where = "round" if h["depth"] > 0 else "launch"
            if h["kind"] == "bd_config":
                per[where] += 1
            elif h["kind"] == "wait":
                waits[where] += 1
    # one round count per launch only if every looped start has the same, readable trip count
    rounds = next(iter(trips)) if len(trips) == 1 and not rounds_unknown else (1 if not trips and not rounds_unknown else None)
    bdcfg = per["launch"] + per["round"] * rounds if rounds is not None else None
    shim_runs = sorted({b["run_b"] for c in channels if c["type"] == "shim" for b in c["bds"]})
    return {"worklocal": str(d), "channels": channels, "rounds": rounds,
            "rounds_unknown": rounds_unknown or (per["round"] > 0 and rounds is None),
            "bd_config": per, "bd_configs_per_launch": bdcfg, "waits": waits,
            "runtime_bds_skipped": skipped, "locked_bds": locked, "shim_run_b": shim_runs,
            "bd_write_path": wp, "has_host_lines": sv is not None}
