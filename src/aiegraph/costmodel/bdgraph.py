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
  start queue  <= START_Q starts outstanding per (tile, channel) between waits, loops unrolled (hw_violations).
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

BD_WRITE_PATHS = ("jit_mmio", "jit_ctrlpkt", "aot_window", "uart_degraded")
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


# Hard HW/runtime limits (Amit, amit/tasks/2026-11-20_des-json-gap/README.md section 2; AIE2PS).
BD_POOL = {"core": 16, "shim": 16, "memtile": 48}   # BDs per tile DMA (shim's 16 shared by its channels)
REPEAT_MAX = 256                                     # start_io repeat_count, 8-bit field (R-1)
START_Q = 4                                          # aie-rt StartQSizeMax: starts outstanding per channel; host blocks on the 5th
WRAP_MAX = 1024                                      # D0..D2 wrap after the eff_dims trim
PLAIN_SHIM_WORDS_MAX = 1023                          # plain shim BD > 1023 words: legacy path, not an AOT "bd" site


def hw_checks(tile_type: Any, bd: Dict[str, Any]) -> List[str]:
    """Per-BD limit violations (strings).  A 1-dim stride-4 BD that collapses to a plain run is exempt from WRAP_MAX."""
    out: List[str] = []
    st, wr = list(bd.get("dim_strides") or []), list(bd.get("dim_wraps") or [])
    n = int(bd.get("len", 0))
    for i, x in enumerate(st):
        if int(x) % 4:
            out.append(f"stride[{i}]={x} not %4")
    eff = len(wr)
    while eff > 1 and wr[eff - 1] <= 1:
        eff -= 1
    plain = eff == 1 and st[:1] == [4] and int(wr[0]) * 4 == n
    if wr and not plain:
        for i, w in enumerate(wr[:min(eff, 3)]):
            if int(w) > WRAP_MAX:
                out.append(f"wrap[{i}]={w} > {WRAP_MAX}")
    return out


def is_plain_large_shim(tile_type: Any, bd: Dict[str, Any]) -> bool:
    """Plain (no multi-dim) shim BD over 1023 words: __Runtime_dma_bd_config takes the legacy path (AXI burst not
    set, not batched in the AOT shim-BD window; Amit des-json-gap section 1)."""
    if tile_type != "shim" or int(bd.get("len", 0)) <= PLAIN_SHIM_WORDS_MAX * 4:
        return False
    st, wr = list(bd.get("dim_strides") or []), list(bd.get("dim_wraps") or [])
    return not st or not wr


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


def start_queue_violations(sv: Dict[str, Any], trip_n: Optional[int]) -> List[str]:
    """Starts outstanding per (tile, channel) between waits, over the host schedule with loops unrolled.

    Amit's bdgraph_extract.py scans the lines once; a start inside a loop whose wait is outside the loop then
    queues one entry per round, so here each loop body runs min(trip, START_Q + 1) times (once if the trip count
    is unknown -- a lower bound).  A loop host_line's own ``trip_count`` (schedule_view, des-json gap s4 ask 1)
    takes precedence over ``trip_n``.  A line belongs to the first channel whose host_lines carry it (as in the extractor).
    """
    seen: Dict[int, Dict[str, Any]] = {}
    for t in sv.get("tiles", []):
        loc = tuple(t.get("loc") or (t.get("col"), t.get("row")))
        for c in t.get("dma_channels", []):
            for h in c.get("host_lines", []):
                seen.setdefault(h["line"], {"line": h["line"], "kind": h["kind"], "trip": h.get("trip_count"),
                                            "key": (loc, f"{c.get('direction')}{c.get('channel')}")})
    seq = [seen[k] for k in sorted(seen) if seen[k]["kind"] in ("startio", "wait", "loop", "loopend")]
    reps = 1 if trip_n is None else max(1, min(int(trip_n), START_Q + 1))
    out: Dict[Any, int] = {}
    worst: Dict[Any, tuple] = {}

    def run(i: int) -> int:                          # executes seq[i:] until the matching loopend; returns its index
        while i < len(seq):
            h = seq[i]
            if h["kind"] == "loop":
                end = i + 1
                n = reps if h["trip"] is None else max(1, min(int(h["trip"]), START_Q + 1))
                for _ in range(n):
                    end = run(i + 1)
                i = end + 1
                continue
            if h["kind"] == "loopend":
                return i
            k = h["key"]
            if h["kind"] == "wait":
                out[k] = 0
            else:
                out[k] = out.get(k, 0) + 1
                if out[k] > START_Q and out[k] > worst.get(k, (0,))[0]:
                    worst[k] = (out[k], h["line"])
            i += 1
        return i

    run(0)
    return [f"line {ln}: {n} starts outstanding on {k[0]} {k[1]} (> {START_Q}, host blocks)"
            for k, (n, ln) in sorted(worst.items(), key=lambda kv: kv[1][1])]


def infer_bd_write_path(d: Path) -> Optional[str]:
    """The shim BD write path the generated host.cc asks for, or None when there is no host.cc.

    __Runtime_ctrl_aot_window( -> "aot_window" (passapitocontrolpacket batches the begin/commit window into it);
    __Runtime_ctrl_shim_bd_begin( -> "jit_ctrlpkt" (#pragma control_plan_shim_bd_ctrl, DfscheduleToApi);
    neither -> "jit_mmio".  This is the intended path: at run time an AOT miss (rt_sb_aot_take == 0) or a
    shim_bd_ctrl fallback writes the BD through RT_SB_CALL instead.  506 designs since 11-10: 405 / 91 / 10, none
    with both calls.
    """
    f = Path(d) / "host.cc"
    if not f.is_file():
        return None
    t = f.read_text(errors="replace")
    if "__Runtime_ctrl_aot_window(" in t:
        return "aot_window"
    if "__Runtime_ctrl_shim_bd_begin(" in t:
        return "jit_ctrlpkt"
    return "jit_mmio"


def features(src: Any) -> Dict[str, Any]:
    """src = worklocal dir (str/Path) or {'worklocal': dir, 'trip_count': int?, 'bd_write_path': str?}.

    bd_write_path not given -> read from worklocal/host.cc (infer_bd_write_path); bd_write_path_src says which.
    """
    spec = dict(src) if isinstance(src, dict) else {"worklocal": src}
    d = Path(spec["worklocal"])
    dfs = _load(d, "dfscheduleprovenancemap.json")
    if dfs is None:
        raise FileNotFoundError(f"no dfscheduleprovenancemap.json in {d}")
    sv = _load(d, "schedule_view.json")
    wp = spec.get("bd_write_path")
    if wp is not None and wp not in BD_WRITE_PATHS:
        raise ValueError(f"bd_write_path {wp!r} not in {BD_WRITE_PATHS}")
    wp_src = "spec" if wp is not None else None
    if wp is None:
        wp = infer_bd_write_path(d)
        wp_src = "host.cc" if wp is not None else None
    ovr = spec.get("trip_count")

    channels, skipped, locked, trips = [], 0, 0, set()
    rounds_unknown = False
    viol: List[str] = []
    plain_large = 0
    for t in dfs.get("tiles", []):
        tloc, ttype = (t.get("col"), t.get("row")), t.get("type")
        ids = {b["bd_id"] for c in t.get("dma_channels", []) for b in c.get("bd_chain", [])
               if isinstance(b.get("bd_id"), int)}
        if ttype in BD_POOL and len(ids) > BD_POOL[ttype]:
            viol.append(f"{ttype}{tloc}: {len(ids)} BDs > pool {BD_POOL[ttype]}")
        if ttype in BD_POOL and any(i < 0 or i >= BD_POOL[ttype] for i in ids):
            viol.append(f"{ttype}{tloc}: bd_id outside 0..{BD_POOL[ttype] - 1}")
        for c in t.get("dma_channels", []):
            bds = []
            for b in c.get("bd_chain", []):
                if not isinstance(b.get("bd_id"), int):
                    skipped += 1
                    continue
                a = (b.get("acquire_lock") or [{}])[0].get("id", -1)
                r = (b.get("release_lock") or [{}])[0].get("id", -1)
                locked += int(a >= 0 and r >= 0)
                viol += [f"{ttype}{tloc} {c.get('direction')}{c.get('channel')} bd{b['bd_id']}: {v}"
                         for v in hw_checks(ttype, b)]
                plain_large += int(is_plain_large_shim(ttype, b))
                bds.append({"bd_id": b["bd_id"], "len": int(b.get("len", 0)), "run_b": run_bytes(b),
                            "packet": bool(b.get("enable_packet"))})
            starts = []
            for s in c.get("start_io", []):
                n = trip(s, ovr)
                rounds_unknown |= n is None
                if n is not None and s.get("inside_scf_for"):
                    trips.add(n)
                rc = int(s.get("repeat_count", 1))
                if not 1 <= rc <= REPEAT_MAX:
                    viol.append(f"{ttype}{tloc} {c.get('direction')}{c.get('channel')}: repeat_count {rc} not in 1..{REPEAT_MAX}")
                starts.append({"repeat_count": rc, "trip": n})
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
    if sv is not None:
        viol += start_queue_violations(sv, ovr if ovr is not None else rounds)
    shim_runs = sorted({b["run_b"] for c in channels if c["type"] == "shim" for b in c["bds"]})
    return {"worklocal": str(d), "channels": channels, "rounds": rounds,
            "rounds_unknown": rounds_unknown or (per["round"] > 0 and rounds is None),
            "bd_config": per, "bd_configs_per_launch": bdcfg, "waits": waits,
            "runtime_bds_skipped": skipped, "locked_bds": locked, "shim_run_b": shim_runs,
            "bd_write_path": wp, "bd_write_path_src": wp_src, "has_host_lines": sv is not None, "ddr_bytes": ddr,
            "hw_violations": viol, "plain_shim_bds_large": plain_large}


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
                         "runs": n, "bytes": b, "run_b": run_bytes(hb) if hb is not None else None,
                         "row_stride_b": row_stride(hb)})
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
# BD write path the board fits (BD_US_FIT, ISSUE_FIT) were measured on.  The 17 kacc kernels write their shim BDs at
# the call as ctrl packets: host.cc wraps them in __Runtime_ctrl_shim_bd_begin/commit (rt_sb_commit), with no
# __Runtime_ctrl_aot_window (Arthur grepped all 17 costmodel_p0/*/aout/worklocal/host.cc, peer ask 11-21 21:5x);
# u_conv_1024x2880x320 does 482 bdcfg calls/launch, rt_bdcfg 1.05 ms/launch (amit/tasks/2026-11-19_costmodel-microbench).
# jit_mmio (direct register writes), aot_window (precomputed shim BDs, rt_sb_aot_take) and uart_degraded (~6 ms/BD,
# amit/tasks/2026-11-20_des-json-gap §3) are different costs, so estimate() lists them as unpriced unless the op/params
# give bd_us explicitly.
BD_US_FIT_PATH = {"palmyra213": "jit_ctrlpkt"}

# Short-DDR-run penalty, per board: {run_b: {"issue": f, "dma": f}, "r_switch": r} -- launch ~ issue + floor x
# f(run_b, regime), regime "dma" when ddr_floor > r_switch x host issue (default r_switch 1), else "issue" (part of
# the slow DMA runs under BD configuration).  A plain number applies to both regimes; run_b not listed -> factor 1.
# palmyra213, nine 32 B-run kernels, f_eff = (launch - issue) / floor, sorted by r = floor / issue
# (xiaojiang/tasks/2026-11-20_runb-p4h, loo2.py):
#   issue  u_lin_1024x320x320 r.15 1.35, 1024x320x960 r.23 1.38, 1024x320x2560 r.27 1.45, 1024x1280x320 r.62 1.50
#   dma    u_lin_1024x1600x320 r.77 1.96, 1024x1920x320 r.93 1.97, 1024x2560x320 r1.24 2.06,
#          u_conv_1024x2880x320 r1.40 1.95, u_lin_1024x5760x320 r2.78 2.00
# The switch is a step between r .62 and .77 (P4i r_switch .70, dma mean 1.99; P4h used 1.0 and missed u_lin_1024x1920x320 by -18%).
# Leave-one-out within regime: max |rel| 3.4%, mean 1.7% (P4g max(issue+floor, 2.67 floor): 14.2% max).
# Same-tree TKW 8 -> 16 A/B (32 -> 64 B runs) confirms the cause on the DMA-bound pair (-32%, -32%).
# Caveat: only one issue-regime kernel above r .3; the step may depend on shape, not r alone.
# P4i (superseded by P4j below, kept for reference / explicit use):
RUN_B_FACTOR_FIT_P4I = {"palmyra213": {32: {"issue": 1.42, "dma": 1.99}, "r_switch": 0.70}}

# P4j (2026-11-20, xiaojiang/tasks/2026-11-20_runb-decomp RESULT5/RESULT6): the P4i "regime" was mostly an error in
# the issue term.  Measured issue (kload+bdcfg+startio+coreen) grows with bytes (bdcfg host glue, startio cache
# flush), so with the board's ISSUE_FIT below one factor per run_b, no regime switch:
#   palmyra213 32 B: wait_io / floor_rd over the 7 kernels at >= 1.5 (mean 1.87; 1.77..2.02).
#   16 B: u_lin_1024x1280x640 only (n=1, in-sample) 2.53.
# Known miss: M=1024 N=320 K<=1280 waits only ~1.37 x floor (u_lin_1024x320x320 +7%, u_lin_1024x1280x320 +16%).
# Over the 10 TKW=8 kernels: |rel| <= 2.4% on 7, +6.8% / +15.8% (low-wait pair), -5.9% (u_lin_512x2560x320).
RUN_B_FACTOR_FIT_P4J = {"palmyra213": {32: 1.87, 16: 2.53}}

# P4k (2026-11-20, xiaojiang/tasks/2026-11-20_runb-stride RESULT7-9): at 32 B runs the factor follows the DDR row
# stride of the read BD (dim_strides[1] in bytes = kp x 2 for the W/X tensor) and nothing else measured -- not M
# (512/1024/2048), not the buffer base address, not rows per stream.  palmyra213 TKW=8, wait_io / floor_rd:
#   fast strides 640 / 1312 / 2560 B (kp 320/656/1280): 1.13..1.41, mean 1.33
#   other strides 1920, 3200, 3840, 5120, 5760, 11520 B: 1.74..2.02, mean 1.87
# EMPIRICAL: the cause is unknown (single-stream and 8-stream open-page bank models and the base offset were all
# rejected), so a stride not in the table takes the conservative slow factor.  Form: {"by_stride": {B: f},
# "default": f} per run_b; plain numbers and {"issue", "dma"} dicts keep working.
# P4l (RESULT11): K=960 padded to kp 1024 (stride 2048 B, KACC_KP_PAD) -> wait/floor 1.25 at M 512/1024/2048
# (kp 960: 1.74..1.77; kp 976 / 1952 B: 1.63, left at the default).  2048 joins the fast class at the class 1.33.
RUN_B_FACTOR_FIT = {"palmyra213": {32: {"by_stride": {640: 1.33, 1312: 1.33, 2048: 1.33, 2560: 1.33}, "default": 1.87},
                                   16: 2.53}}

# Host issue time per launch, per board (P4j): issue_ms = bd_ms x bd_configs_per_launch + rd_frac x ddr_floor rd term
# (ms) + c_ms.  palmyra213: least squares over nine 32 B-run kernels in three BD-count families (216/480/864 configs
# per launch), max residual 39 us; the bare (cfg+2) x 5.67 us form was 0.22-0.31 ms low at M=2048 / M=512.
# Only used for the board fit (an op or params bd_us keeps the plain cfg x bd_us form).
ISSUE_FIT = {"palmyra213": {"bd_ms": 5.31e-3, "rd_frac": 0.160, "c_ms": 0.111}}


def issue_fit_ms(fit: Dict[str, Any], bd_configs: int, floor: Optional[Dict[str, Any]],
                 params: Dict[str, Any]) -> Optional[float]:
    """P4j host issue time: bd_ms x configs + rd_frac x floor rd (ms) + c_ms; None without a floor rd term."""
    if floor is None or "rd" not in floor.get("terms", {}):
        return None
    rd_ms = floor["terms"]["rd"] / (float(params["aie_clk_mhz"]) * 1e3)
    return float(fit["bd_ms"]) * bd_configs + float(fit["rd_frac"]) * rd_ms + float(fit["c_ms"])


def row_stride(bd: Optional[Dict[str, Any]]) -> Optional[int]:
    """DDR row stride (bytes) of a strided BD = its second dim stride; None for a plain contiguous BD."""
    if bd is None:
        return None
    st, wr = list(bd.get("dim_strides") or []), list(bd.get("dim_wraps") or [])
    if len(st) < 2 or len(wr) < 2 or wr[1] <= 1 or run_bytes(bd) >= int(bd.get("len", 0)):
        return None
    return int(st[1])


def _runb_factor(v: Any, regime: str, stride: Optional[int] = None) -> float:
    if v is None:
        return 1.0
    if isinstance(v, dict):
        if "by_stride" in v:
            return float(v["by_stride"].get(stride, v.get("default", 1.0)))
        return float(v.get(regime, 1.0))
    return float(v)


def ddr_runb(floor: Optional[Dict[str, Any]], ddr: Optional[Dict[str, Any]],
             factors: Optional[Dict[int, Any]], issue_ms: Optional[float] = None) -> Optional[Dict[str, Any]]:
    """Run-length-adjusted DDR time (P4h; P4i adds r_switch; P4k picks a factor by the channel's row_stride_b): ms = floor x the MM2S-byte-weighted factor of each read channel's head-BD
    run_b, the factor picked by regime ("dma" when floor > factors["r_switch"] (default 1) x issue_ms, else "issue";
    "dma" without issue_ms).
    With ``issue_ms`` also launch_ms = issue + ms.  An estimate, not a bound -- report only, like ddr_floor."""
    if floor is None or ddr is None or not factors:
        return None
    rd = [c for c in ddr["channels"] if c["bytes"] and str(c["dir"]).upper() == "MM2S"]
    tot = sum(c["bytes"] for c in rd)
    if not tot:
        return None
    r_switch = float(factors.get("r_switch", 1.0))
    regime = "dma" if issue_ms is None or floor["ms"] > r_switch * float(issue_ms) else "issue"
    fac = sum(c["bytes"] * _runb_factor(factors.get(c.get("run_b")), regime, c.get("row_stride_b")) for c in rd) / tot
    out = {"ms": floor["ms"] * fac, "factor": fac, "regime": regime}
    if issue_ms is not None:
        out["launch_ms"] = float(issue_ms) + out["ms"]
    return out


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
