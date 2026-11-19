"""Discrete-event model of ONE gemm_kacc launch on the 4x4 AIE mesh (AIE side, segment K of the cost layer).

Predicted quantity: the host time spent in ``__Runtime_wait_io`` per launch (``rt_wait_io`` of
``LAUNCH split_detail``, src/aiegraph/runtime/main_launch_line.c:45) = the AIE-side time the host's own BD issue
does not hide.  Host glue / launch overhead are NOT predicted here (segment L); the measured host BD-issue cost
per BD enters only as the unit of the issue/AIE overlap (``op['host']['bd_us']``, see host_from_split).

Launch structure (generated host.cc of every kacc instance, aiehlc DF schedule):
  * R = ntile rounds; per round the host issues 24 BDs (4 b-port BDs, one per mesh column and broadcast to its 4
    cores; then per mesh row 1 a-port BD + 4 c-port BDs), then waits on all 12 io events of the round.
  * per round and core: nkc K-chunks; chunk j = a rows 16 x kch bf16 + b rows 16*S x kch bf16 in 2-deep
    (ping-pong) windows, compute, release; one fp32 c-window commit (16 x 16*S x 4 B) after the last chunk.
  * broadcast + 2-deep windows => the 16 cores advance in lock step; one shim column per mesh column.

Event model per round (cycles at aie_clk_mhz): chunk j arrives at max(arrival j-1, release of chunk j-2) +
dma_chunk, starts at max(arrival, end of chunk j-1), takes 2 lock acquires + exposed compute; the round ends with
the c commit drain.  The host waits max(0, round - overlap) where overlap = overlap_bd host BD slots of the
round's own issue that the AIE work hides.

Terms:
  compute   core_cyc (ISS, one core, one launch: K1b) + chunk_ovh_cyc * R*nkc + round_ovh_cyc * R
            (window acquire/release + loop/commit overhead the ISS harness does not see; chunks/rounds as in
            board_compute.json: chunks = NT*NKC, rounds = NT output windows)
  dma_chunk max(tile stream (a+b)/nkc / dma_b_per_cyc, DDR round time / nkc) + bd_switch_cyc,
            + dma_first_byte_cyc for the first chunk.  DDR round time:
            * per burst (params carry ddr_burst_cyc0 + ddr_burst_b_per_cyc, run_microbench.sh --section ddr):
              bursts = mesh-unique bytes / run spread over 4 shim columns, each burst ddr_burst_cyc0 + run /
              ddr_burst_b_per_cyc AIE cycles (strided-burst microbenchmark), times ddr_bw_scale (calibrated:
              achieved / independent-column burst time of the concurrent 4-column kacc traffic); no long-run cap.
            * else bytes-only: bytes / ddr_eff, ddr_eff = ddr_bw_scale * ddr_rd_gbps/clk * run/(run+ddr_run0_b).
            run = kch*2 B = contiguous run of the multidim shim BD (d0_wrap = tkw words)
  lock      lock_acq_cyc per window acquire (2 per chunk + 1 per commit)
  stall     core idle inside the round (waiting for data) -- the board's lock_stall + stream_stall also counts the
            idle time while the host is still issuing (reported, not fitted)
  wait      WAIT WINDOW: the host enters __Runtime_wait_io only after its last BD of the round, while the b-port
            stream (BDs issued first) and the cores already run.  Only the DES timeline after the wait start is
            predicted: per round max(0, round + round_fixed_cyc - overlap - b_overlap_cyc * b/(a+b)), b/(a+b) = the
            b-port share of the round's bytes (S/(S+1)); sum over rounds.
Calibrated keys (script/perf/costmodel_fit.py, defaults = uncalibrated physics): CAL_DEFAULTS.  The palmyra213
calibration (K2c) uses 6: chunk_ovh_cyc (compute), ddr_bw_scale + ddr_run0_b (bytes-only DDR, ddr_model="bytes"),
b_prefetch_b, round_fixed_cyc and compute_exposed.  compute_exposed calibrates to ~0: inside the host's wait window
the round is DDR-tail-limited; the board PERFIO split (R=1 yolo launches) shows >=88 % of the core compute runs
before the host enters __Runtime_wait_io, i.e. hidden behind the host BD issue.
"""
from __future__ import annotations

from typing import Any, Dict

CAL_DEFAULTS = {
    "chunk_ovh_cyc": 0.0,    # core cycles per K-chunk not in the ISS count
    "round_ovh_cyc": 0.0,    # core cycles per round (/S) not in the ISS count
    "compute_exposed": 1.0,  # fraction of the core compute on the round's critical path
    "ddr_bw_scale": 1.0,     # achieved / microbenchmark DDR read bandwidth for long runs
    "ddr_run0_b": 0.0,       # per-run DDR overhead in byte-equivalents
    "overlap_bd": 0.0,       # host BD slots of a round's issue that hide its AIE work
    "b_prefetch_b": 0.0,     # mesh bytes of the b-port stream that run ahead of the cores while the host is
                             # still issuing the round (b BDs go first; shim/stream FIFO + window slack)
    "round_fixed_cyc": 0.0,  # fixed per-round latency (DMA start / event round trip)
    "wait_call_cyc": 0.0,    # host cost per __Runtime_wait_io call (structural count per round)
    "b_overlap_cyc": 0.0,    # wait-window start offset: AIE cycles of the round that run before the host's wait
                             # starts, per unit b-port byte share (b streams while the host still issues a/c BDs)
}
BDS_PER_ROUND = 24  # 4 b + 4 * (1 a + 4 c)


def _p(params: Dict[str, Any], key: str) -> float:
    return float(params[key]) if key in params else float(CAL_DEFAULTS[key])


def geometry(op: Dict[str, Any]) -> Dict[str, Any]:
    """Per-round / per-chunk byte counts of a kacc instance (op['instance'] as in aie2ps_kernel_cycles.json)."""
    i = op["instance"]
    R, S, k, nkc, kch = int(i["ntile"]), int(i["S"]), int(i["k"]), int(i["nkc"]), int(i["kch"])
    a_b = 16 * k * 2            # a-port bytes per core per round
    b_b = S * 16 * k * 2        # b-port bytes per core per round
    c_b = 16 * S * 16 * 4       # c-port (fp32) bytes per core per round
    return {"R": R, "S": S, "k": k, "nkc": nkc, "kch": kch, "a_b": a_b, "b_b": b_b, "c_b": c_b,
            "chunks": R * nkc, "run_b": kch * 2,
            "ddr_rd_round_b": 4 * (a_b + b_b), "ddr_wr_round_b": 16 * c_b}


def compute_cycles(op: Dict[str, Any], params: Dict[str, Any], cycles: Dict[str, Any]) -> float:
    """DES compute term (core busy, no stalls) per launch, AIE cycles, slowest core.  Compared on the board with
    max-tile active_cyc - lock_stall - stream_stall."""
    g = geometry(op)
    return float(cycles["core_cyc"]) + _p(params, "chunk_ovh_cyc") * g["chunks"] + \
        _p(params, "round_ovh_cyc") * g["R"]


def ddr_eff_bpc(op: Dict[str, Any], params: Dict[str, Any]) -> float:
    """Achieved DDR read bytes per AIE cycle for this instance's shim BD run length."""
    g = geometry(op)
    bpc = float(params["ddr_rd_gbps"]) * 1e3 / float(params["aie_clk_mhz"])  # GB/s -> B per AIE cycle
    run = g["run_b"]
    return _p(params, "ddr_bw_scale") * bpc * run / (run + _p(params, "ddr_run0_b"))


def burst_model(params: Dict[str, Any]) -> bool:
    """Per-burst DDR model only if the params carry the burst microbenchmark and the calibration does not select the
    bytes-only model (cal key ddr_model="bytes": the single-column strided-burst time does not transfer to the
    concurrent 4-column kacc traffic on palmyra213, K2c: maxrel 0.309 per-burst vs 0.190 bytes-only)."""
    return params.get("ddr_model", "burst") != "bytes" and "ddr_burst_cyc0" in params and \
        "ddr_burst_b_per_cyc" in params


def ddr_burst_cyc(op: Dict[str, Any], params: Dict[str, Any]) -> float:
    """AIE cycles of one shim burst of this instance's run length (strided-burst microbenchmark, one column)."""
    run = geometry(op)["run_b"]
    return float(params["ddr_burst_cyc0"]) + run / float(params["ddr_burst_b_per_cyc"])


def ddr_read_cyc(op: Dict[str, Any], params: Dict[str, Any], nbytes: float) -> float:
    """AIE cycles to read nbytes mesh-unique DDR bytes of this instance (per burst if the params carry the burst
    microbenchmark, else bytes-only)."""
    if burst_model(params):
        run = geometry(op)["run_b"]
        return nbytes / run / 4.0 * ddr_burst_cyc(op, params) * _p(params, "ddr_bw_scale")
    return nbytes / ddr_eff_bpc(op, params)


def predict_launch(op: Dict[str, Any], params: Dict[str, Any], cycles: Dict[str, Any]) -> Dict[str, Any]:
    """Simulate one launch.  op = {'instance': {...}, 'host': {'bd_us': ...}}; params = palmyra213.json (+ cal);
    cycles = {'core_cyc': ...}.  Returns ms per launch:
    dict(compute_ms, dma_ms, lock_ms, stall_ms, wait_ms, bound, aie_ms, ddr_ms, rounds)."""
    g = geometry(op)
    clk_mhz = float(params["aie_clk_mhz"])
    host = op.get("host") or {}
    bd_cyc = float(host.get("bd_us", params.get("host_bd_us", 5.0))) * clk_mhz
    R, nkc = g["R"], g["nkc"]

    comp_total = compute_cycles(op, params, cycles)
    lock_acq = float(params["lock_acq_cyc"])
    comp_chunk = max(0.0, comp_total / (R * nkc) - 2 * lock_acq) * _p(params, "compute_exposed")

    dma_bpc = float(params["dma_b_per_cyc"])
    first = float(params["dma_first_byte_cyc"])
    ddr_eff = ddr_eff_bpc(op, params)
    ddr_wr_bpc = float(params["ddr_wr_gbps"]) * 1e3 / clk_mhz
    # b-port BDs are issued first (slots 0-3): while the host issues the other 20 BDs of the round, up to
    # b_prefetch_b bytes of b stream ahead; the rest (all of a + the remaining b) is paced by the 2-deep windows
    b_round = 4 * g["b_b"]
    pre = min(b_round, _p(params, "b_prefetch_b"), ddr_eff * (BDS_PER_ROUND - 4) * bd_cyc)
    rest_b = g["ddr_rd_round_b"] - pre
    tile_chunk = (g["a_b"] + g["b_b"]) / nkc / dma_bpc
    ddr_chunk = ddr_read_cyc(op, params, rest_b) / nkc
    dma_chunk = max(tile_chunk, ddr_chunk) + float(params["bd_switch_cyc"])
    # c commit: the core hands its fp32 window to the tile S2MM->shim path; the DDR write of the 16 windows overlaps
    # the next round's issue (output events are waited, but the write queue drains at ddr_wr while the host issues)
    c_drain = g["c_b"] / dma_bpc + first + lock_acq
    overlap = _p(params, "overlap_bd") * bd_cyc + \
        _p(params, "b_overlap_cyc") * g["b_b"] / (g["a_b"] + g["b_b"])
    wpr = float(host.get("waits_per_launch", 12 * R)) / R
    fixed = _p(params, "round_fixed_cyc") + _p(params, "wait_call_cyc") * wpr

    # one round after the host's last BD of the round (all rounds identical: the host serialises them by
    # waiting on the round's events); chunk j arrives when the window of chunk j-2 is released
    done = [0.0, 0.0]
    arr = first
    t_done = 0.0
    idle = 0.0
    dma_limited = 0
    for _j in range(nkc):
        arr = max(arr, done[-2]) + dma_chunk
        start = max(arr, t_done)
        if arr > t_done:
            dma_limited += 1
            idle += arr - t_done
        t_done = start + 2 * lock_acq + comp_chunk
        done.append(t_done)
    t_round = t_done + c_drain
    wait_round = max(0.0, t_round + fixed - overlap)

    to_ms = 1.0 / (clk_mhz * 1e3)
    return {
        "compute_ms": comp_total * to_ms,
        "dma_ms": R * (nkc * dma_chunk + first) * to_ms,
        "lock_ms": R * (2 * nkc + 1) * lock_acq * to_ms,
        "stall_ms": R * (idle + c_drain) * to_ms,
        "wait_ms": R * wait_round * to_ms,
        "aie_ms": R * t_round * to_ms,
        "ddr_ms": R * ddr_read_cyc(op, params, g["ddr_rd_round_b"]) * to_ms,
        "rounds": R,
        "bound": "ddr" if dma_limited * 2 > nkc else "compute",
    }


def host_from_split(split_detail: Dict[str, Any], rounds: int | None = None) -> Dict[str, Any]:
    """Host BD-issue cost of a measured launch (an INPUT of the DES; segment L later predicts it):
    host time per BD config = (rt_bdcfg + rt_startio + host_glue_bdcfg) / BD configs per launch.
    rt_wait_io (the predicted quantity) is never read here."""
    o = split_detail["other"]
    L = float(o.get("launches") or split_detail["calls"].get("coreen") or 1)
    bd_per_launch = split_detail["calls"]["bdcfg"] / L
    rt, hs = split_detail["rt"], split_detail["host"]
    host_ms = rt.get("bdcfg", 0.0) + rt.get("startio", 0.0) + hs.get("glue_bdcfg", 0.0)
    d = {"bd_us": host_ms * 1e3 / bd_per_launch, "bd_per_launch": bd_per_launch, "issue_ms": host_ms}
    if rounds is not None:
        d["bds_pre"] = max(0, int(round(bd_per_launch - BDS_PER_ROUND * rounds)))
    return d
