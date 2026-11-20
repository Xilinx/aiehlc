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

  ddr_bytes    per shim channel and launch = sum over its host startio lines of
               chain_bytes(createio bd_id) x R x (trip_count if the startio is inside the loop).
               R = 4th arg of ``__Runtime_startio(dev, io, bd, R)`` (else start_io.repeat_count); one start runs the
               BD chain R times and every run moves the full ``len`` (aie-rt XAie_DmaChannelSetStartQueue writes
               R-1 to RptCount; per Amit, peer:1120010346-95530d).  chain_bytes follows next_bd from the createio
               BD until -1 or a cycle.  The iter dim changes the address, not the byte count.
               ``ddr_floor`` turns these bytes into a report-only lower bound (board stream + DDR column points);
               ``estimate`` reports it as "ddr" and keeps it out of total_ms, like the kacc form's DES ddr term.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

BD_WRITE_PATHS = ("jit_mmio", "aot_window", "uart_degraded")
_RANGE = re.compile(r"^\s*(-?\d+)\s*\.\.\s*(-?\d+)\s*$")
_STARTIO_R = re.compile(r"__Runtime_startio\([^,]+,[^,]+,[^,]+,\s*(\d+)\s*\)")


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


def chain_bytes(bds: List[Dict[str, Any]], head: Any) -> Optional[int]:
    """Bytes of one run of the BD chain that starts at ``head`` (next_bd until -1 or a repeat)."""
    by = {b["bd_id"]: b for b in bds if isinstance(b.get("bd_id"), int)}
    if head not in by:
        return None
    n, seen, cur = 0, set(), head
    while cur in by and cur not in seen:
        seen.add(cur)
        n += int(by[cur].get("len", 0))
        cur = by[cur].get("next_bd", -1)
    return n


def _line_depth(sv: Dict[str, Any]) -> Dict[int, int]:
    return {h["line"]: h["depth"] for h in _host_lines(sv)}


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
    ddr = _ddr_bytes(dfs, sv, ovr, rounds) if sv is not None else None
    shim_runs = sorted({b["run_b"] for c in channels if c["type"] == "shim" for b in c["bds"]})
    return {"worklocal": str(d), "channels": channels, "rounds": rounds,
            "rounds_unknown": rounds_unknown or (per["round"] > 0 and rounds is None),
            "bd_config": per, "bd_configs_per_launch": bdcfg, "waits": waits,
            "runtime_bds_skipped": skipped, "locked_bds": locked, "shim_run_b": shim_runs,
            "bd_write_path": wp, "has_host_lines": sv is not None, "ddr_bytes": ddr}


def _ddr_bytes(dfs: Dict[str, Any], sv: Dict[str, Any], ovr: Optional[int], rounds: Optional[int]) -> Dict[str, Any]:
    """DDR bytes per launch on the shim channels: {'mm2s': int|None, 's2mm': int|None, 'channels': [...]}."""
    depth = _line_depth(sv)
    chains = {}
    for t in dfs.get("tiles", []):
        if t.get("type") == "shim":
            for c in t.get("dma_channels", []):
                chains[(t.get("col"), t.get("row"), c.get("direction"), c.get("channel"))] = c
    tot: Dict[str, Optional[int]] = {"mm2s": 0, "s2mm": 0}
    rows = []
    for t in sv.get("tiles", []):
        loc = tuple(t.get("loc") or (None, None))
        for c in t.get("dma_channels", []):
            key = (loc[0], loc[1], c.get("direction"), c.get("channel"))
            dc = chains.get(key)
            if dc is None:
                continue
            hl = c.get("host_lines", [])
            head = next((h.get("bd_id") for h in hl if h["kind"] == "createio"), None)
            per_run = chain_bytes(dc.get("bd_chain", []), head)
            r_default = int((dc.get("start_io") or [{}])[0].get("repeat_count", 1))
            n, ok = 0, per_run is not None
            for h in hl:
                if h["kind"] != "startio":
                    continue
                m = _STARTIO_R.search(h.get("code", ""))
                r = int(m.group(1)) if m else r_default
                k = 1 if depth.get(h["line"], 0) == 0 else (ovr if ovr is not None else rounds)
                if k is None:
                    ok = False
                    continue
                n += r * int(k)
            b = n * per_run if ok else None
            d = "mm2s" if str(key[2]).upper() == "MM2S" else "s2mm"
            hb = next((x for x in dc.get("bd_chain", []) if x.get("bd_id") == head), None)
            rows.append({"tile": loc, "dir": key[2], "ch": key[3], "head_bd": head, "run_bytes": per_run,
                         "runs": n, "bytes": b, "run_b": run_bytes(hb) if hb is not None else None})
            tot[d] = None if (b is None or tot[d] is None) else tot[d] + b
    return {"mm2s": tot["mm2s"], "s2mm": tot["s2mm"], "channels": rows}


def _interp_cols(points: List[List[float]], n: int) -> float:
    """GB/s at n active shim columns from the board's [[cols, GB/s], ...] points (clamped at both ends)."""
    pts = sorted((int(c), float(g)) for c, g in points)
    for c, g in pts:
        if c >= n:
            return g
    return pts[-1][1]


# Host issue cost per BD config (us) for the bdgraph form, per board.  palmyra213: median of
# (bdcfg+startio ms) / measured bdcfg calls over Arthur's 17 kacc kernels (k_pred_vs_meas.tsv, profiling on);
# leave-one-out median |rel| of launch ~ issue + ddr_floor = 7.4%, same as in-sample.  Conv kernels run
# 6.5-9.5 us (issue blocks when DMA per config is slower); see xiaojiang/tasks/2026-11-20_ddr-floor-vs-meas.
BD_US_FIT = {"palmyra213": 5.67}

# Short-DDR-run penalty on the read floor, per board: {run_b: factor}.  palmyra213: the 4 u_lin_1024 + u_conv_1024
# kernels read with 32 B runs and their (launch - issue) / ddr_floor is 1.27-1.83 (64/128 B kernels 0.84-1.30);
# leave-one-out median factor 1.42-1.49 -> 1.45.  LOO launch error max |rel| 0.358 -> 0.199, mean 7.4% -> 4.7%.
# Caveat: run_b == 32 coincides with M == 1024 in that corpus (xiaojiang/tasks/2026-11-20_ddr-floor-vs-meas/runb_loo.txt).
RUN_B_FACTOR_FIT = {"palmyra213": {32: 1.45}}


def ddr_runb(floor: Optional[Dict[str, Any]], ddr: Optional[Dict[str, Any]],
             factors: Optional[Dict[int, float]]) -> Optional[Dict[str, Any]]:
    """Run-length-adjusted DDR time: floor ms x the MM2S-byte-weighted factor of each read channel's head-BD run_b
    (factor 1 for run_b not in ``factors``).  An estimate, not a bound -- report only, like ddr_floor."""
    if floor is None or ddr is None or not factors:
        return None
    rd = [c for c in ddr["channels"] if c["bytes"] and str(c["dir"]).upper() == "MM2S"]
    tot = sum(c["bytes"] for c in rd)
    if not tot:
        return None
    fac = sum(c["bytes"] * float(factors.get(c.get("run_b"), 1.0)) for c in rd) / tot
    return {"ms": floor["ms"] * fac, "factor": fac}


def ddr_floor(ddr: Optional[Dict[str, Any]], params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Lower bound in AIE cycles for moving one launch's shim bytes -- report only, never added to total_ms.

    floor = max(stream, rd, wr):
      stream = max over shim channels of bytes / dma_b_per_cyc            (one 64-bit stream port per channel)
      rd     = MM2S bytes / (ddr_rd_cols_points at the number of active MM2S shim columns, GB/s -> B/cyc)
      wr     = S2MM bytes / (ddr_wr_cols_points at the number of active S2MM shim columns)
    It ignores the per-task host cost (Arthur vek385-3: ~14.7k cyc per DMA task, see bd issue / wait terms),
    run-length effects (run_b) and overlap with compute, so the launch can only be slower than this.
    None when bytes are unknown or the board params lack the points."""
    need = ("aie_clk_mhz", "dma_b_per_cyc", "ddr_rd_cols_points", "ddr_wr_cols_points")
    if ddr is None or ddr.get("mm2s") is None or ddr.get("s2mm") is None or any(k not in params for k in need):
        return None
    clk = float(params["aie_clk_mhz"])
    stream = max([c["bytes"] for c in ddr["channels"] if c["bytes"]] or [0]) / float(params["dma_b_per_cyc"])
    terms = {"stream": stream}
    for d, key in (("mm2s", "ddr_rd_cols_points"), ("s2mm", "ddr_wr_cols_points")):
        cols = {c["tile"][0] for c in ddr["channels"] if c["bytes"] and str(c["dir"]).lower() == d}
        if ddr[d] and cols:
            bpc = _interp_cols(params[key], len(cols)) * 1e3 / clk
            terms["rd" if d == "mm2s" else "wr"] = ddr[d] / bpc
    bound = max(terms, key=terms.get)
    cyc = terms[bound]
    return {"cyc": cyc, "ms": cyc / (clk * 1e3), "bound": bound, "terms": terms}
