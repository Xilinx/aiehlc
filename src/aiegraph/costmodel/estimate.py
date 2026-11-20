"""``aiegraph.costmodel.estimate`` -- the one public entry of the AIEHLC local cost layer (plan P4a).

    estimate(group, board="palmyra213", params=None, cycles=None) -> dict

Input schema (settled in the cost channel: Steve's form, field ``recipe`` not ``dtype``):

    group = {"ops": [op, ...], "launches": 1}         # or a bare list of ops
    op    = {"kernel": "u_lin_1024x320x320",          # name in aie2ps_kernel_cycles.json (optional if recipe+MKN match)
             "recipe": "kacc_gemm_bf16",
             "M": 1024, "K": 320, "N": 320,
             "bytes_in": int, "bytes_out": int,       # carried through; the K model derives bytes from geometry
             "resident": False,                       # carried through (not priced yet, see "unpriced")
             # optional overrides / extra inputs
             "instance": {...},                       # kacc geometry (ntile,S,k,nkc,kch,...) -- else looked up
             "core_cyc": int,                         # ISS cycles -- else looked up
             "host": {"bd_us": float},                # host BD-issue cost per BD (segment L input)
             "l": {"cfg": "off|off-cold|aot|aot-kco", "calls": {...}, "words": {...}},  # L features, per launch
             "bdgraph": {...}}                        # RESERVED: BD/port/lock graph exported from the build JSON
                                                      # (Amit's des-json gap table); NotImplementedError for now

Output (ms per group execution; per-op rows in "ops"):

    {"kload", "cfg", "start", "wait", "glue", "compute", "ddr", "total_ms", "bound", "launches",
     "l_ms", "unpriced": [...], "ops": [...]}

    wait     = segment K, des.predict_launch wait_ms (host time in __Runtime_wait_io)       -- priced
    compute  = DES core compute (reported; mostly hidden behind host BD issue on palmyra213) -- report only
    ddr      = DES DDR read time of the launch's mesh-unique bytes                           -- report only
    l_ms     = segment L, launch_l.predict_l (kload + core_reset + coreen + rt_bdcfg + rt_startio), only if the
               op carries "l"; its per-term split (kload/cfg/start) is not fitted separately -> those stay None
    glue     = segment N (host glue), M2 -- None
    total_ms = sum of the priced terms (wait + l_ms) x launches; terms that could not be priced are listed in
               "unpriced" so a caller never mistakes a partial sum for the full launch time.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import des

PARAMS_DIR = Path(__file__).resolve().parents[3] / "script" / "perf" / "params"
CYCLES = "/scratch/staff/huaj/aiengineers/maxwell/tasks/2026-11-19_kernel-cycles/aie2ps_kernel_cycles.json"
KACC_PREFIX = "kacc_"  # kacc_gemm_bf16, kacc_conv3x3_bf16, ...: same 4x4 kacc instance (des.geometry)
_CAL_EXTRA = ("ddr_model",)


def board_params(board: str = "palmyra213", params_dir: Optional[Path] = None) -> Dict[str, Any]:
    """<board>.json (microbench) + <board>_cal.json (K calibration: CAL_DEFAULTS keys and ddr_model)."""
    d = Path(params_dir) if params_dir else PARAMS_DIR
    base = d / f"{board}.json"
    if not base.exists():
        raise FileNotFoundError(f"no params for board {board!r}: {base}")
    p = json.loads(base.read_text())
    cal = d / f"{board}_cal.json"
    if cal.exists():
        c = json.loads(cal.read_text())
        p.update({k: v for k, v in c.items() if k in des.CAL_DEFAULTS or k in _CAL_EXTRA})
    return p


def load_cycles(path: Optional[str] = None) -> List[Dict[str, Any]]:
    d = json.loads(Path(path or CYCLES).read_text())
    return d["kernels"] if isinstance(d, dict) else d


def _lookup(op: Dict[str, Any], table: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    name = op.get("kernel")
    for k in table:
        if name and k.get("kernel") == name:
            return k
    for k in table:
        if k.get("recipe") == op.get("recipe") and all(k.get(x) == op.get(x) for x in ("M", "K", "N")):
            return k
    return None


def _ops(group: Any) -> List[Dict[str, Any]]:
    if isinstance(group, dict):
        ops = group.get("ops")
        if ops is None:
            raise ValueError("group must be a list of ops or a dict with 'ops'")
        return list(ops)
    return list(group)


def estimate_op(op: Dict[str, Any], params: Dict[str, Any],
                table: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    if "bdgraph" in op:
        raise NotImplementedError("bdgraph input (BD/port/lock graph from the build JSON) is reserved until the "
                                  "des-json gap table lands (amit/tasks/2026-11-20_des-json-gap/)")
    recipe = op.get("recipe")
    if not str(recipe).startswith(KACC_PREFIX):
        raise NotImplementedError(f"recipe {recipe!r} has no K model yet (only {KACC_PREFIX}* instances)")
    inst, core = op.get("instance"), op.get("core_cyc")
    if inst is None or core is None:
        k = _lookup(op, table if table is not None else load_cycles())
        if k is None:
            raise KeyError(f"no kacc instance / core_cyc for op {op.get('kernel') or (op.get('M'), op.get('K'), op.get('N'))}"
                           "; pass 'instance' and 'core_cyc' or add it to aie2ps_kernel_cycles.json")
        inst = inst if inst is not None else k["instance"]
        core = core if core is not None else k["core_cyc"]
    k_op = {"instance": inst, "host": op.get("host") or {}}
    r = des.predict_launch(k_op, params, {"core_cyc": core})
    unpriced = ["glue"]
    l_ms = None
    if op.get("l"):
        from .launch_l import predict_l
        lf = op["l"]
        l_ms = predict_l(lf.get("calls", {}), lf.get("words", {}), lf.get("cfg", "off"))
    else:
        unpriced.append("l")
    if op.get("resident"):
        unpriced.append("resident")
    if "bd_us" not in (op.get("host") or {}) and "host_bd_us" not in params:
        unpriced.append("host_bd_us(default 5.0)")
    return {"kernel": op.get("kernel"), "recipe": recipe, "wait": r["wait_ms"], "compute": r["compute_ms"],
            "ddr": r["ddr_ms"], "l_ms": l_ms, "bound": r["bound"], "rounds": r["rounds"],
            "bytes_in": op.get("bytes_in"), "bytes_out": op.get("bytes_out"), "unpriced": unpriced}


def estimate(group: Any, board: str = "palmyra213", params: Optional[Dict[str, Any]] = None,
             cycles: Optional[str] = None) -> Dict[str, Any]:
    """Predicted ms of one execution of ``group`` (see module docstring for the schema)."""
    prm = params if params is not None else board_params(board)
    table = load_cycles(cycles)
    ops = _ops(group)
    launches = int(group.get("launches", 1)) if isinstance(group, dict) else 1
    rows = [estimate_op(op, prm, table) for op in ops]
    wait = sum(r["wait"] for r in rows)
    l_vals = [r["l_ms"] for r in rows if r["l_ms"] is not None]
    l_ms = sum(l_vals) if l_vals else None
    unpriced = sorted({u for r in rows for u in r["unpriced"]})
    bounds = {r["bound"] for r in rows}
    return {"kload": None, "cfg": None, "start": None, "glue": None,
            "wait": wait * launches, "compute": sum(r["compute"] for r in rows) * launches,
            "ddr": sum(r["ddr"] for r in rows) * launches, "l_ms": l_ms * launches if l_ms is not None else None,
            "total_ms": (wait + (l_ms or 0.0)) * launches,
            "bound": bounds.pop() if len(bounds) == 1 else "mixed", "launches": launches,
            "board": board, "unpriced": unpriced, "ops": rows}
