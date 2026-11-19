"""Roofline flag of one kacc launch: compute-bound vs DDR-bound (AIE side, segment K).

intensity = MACs per launch / mesh-unique DDR bytes per launch (des.geometry); ridge = peak MAC/s of the 16 cores
at the ISS-measured MAC/cycle / achieved DDR B/s (des.ddr_read_cyc: per burst when the params carry the burst microbenchmark).  bound = 'compute' when intensity >= ridge.
"""
from __future__ import annotations

from typing import Any, Dict

from .des import compute_cycles, ddr_read_cyc, geometry


def roofline(op: Dict[str, Any], params: Dict[str, Any], cycles: Dict[str, Any]) -> Dict[str, Any]:
    g = geometry(op)
    macs = float(cycles.get("core_macs", 0.0)) * 16.0  # core_macs is per core
    ddr_b = g["ddr_rd_round_b"] * g["R"] + g["ddr_wr_round_b"] * g["R"]
    comp_cyc = compute_cycles(op, params, cycles)
    t_comp = comp_cyc
    rd_b = g["ddr_rd_round_b"] * g["R"]
    t_ddr = ddr_read_cyc(op, params, rd_b)
    intensity = macs / ddr_b if ddr_b else 0.0
    ridge = (macs / comp_cyc) / (rd_b / t_ddr) if comp_cyc and macs and t_ddr else 0.0
    return {"intensity_mac_per_b": intensity, "ridge_mac_per_b": ridge, "t_compute_cyc": t_comp,
            "t_ddr_cyc": t_ddr, "bound": "compute" if t_comp >= t_ddr else "ddr"}
