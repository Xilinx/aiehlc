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
             "bdgraph": {"worklocal": dir,            # build JSON dir (dfscheduleprovenancemap + schedule_view)
                         "bd_write_path": "jit_mmio|aot_window|uart_degraded",  # required to price issue_ms;
                                                      #   the board fit only prices its own path (palmyra213: jit_mmio)
                         "trip_count": int}}          # optional; else start_io.loop_range (see bdgraph.py)
                                                      # -> counts only + issue_ms + ddr floor (report only,
                                                      #    bdgraph.ddr_floor); K terms listed in "unpriced"

Output (ms per group execution; per-op rows in "ops"):

    {"kload", "cfg", "start", "wait", "glue", "compute", "ddr", "total_ms", "bound", "launches",
     "l_ms", "unpriced": [...], "ops": [...]}

    wait     = segment K, des.predict_launch wait_ms (host time in __Runtime_wait_io)       -- priced
    compute  = DES core compute (reported; mostly hidden behind host BD issue on palmyra213) -- report only
    ddr      = DES DDR read time of the launch's mesh-unique bytes                           -- report only
    l_ms     = segment L, launch_l.predict_l (kload + core_reset + coreen + rt_bdcfg + rt_startio), only if the
               op carries "l"; its per-term split (kload/cfg/start) is not fitted separately -> those stay None
    glue     = segment N (host glue), M2 -- None
    total_ms = sum of the priced terms (wait + l_ms + issue_ms) x launches; terms that could not be priced are listed in
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
        return estimate_bdgraph(op, params)
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


def estimate_bdgraph(op: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    """bdgraph form: structural counts from the build JSON (bdgraph.features) + the host BD-issue time
    (bd_configs x bd_us) when the BD write path is known.  K-segment terms are not priced yet."""
    from . import bdgraph
    f = bdgraph.features(op["bdgraph"])
    unpriced = ["glue", "l", "wait(bdgraph K model)"]
    floor = bdgraph.ddr_floor(f["ddr_bytes"], params)
    if floor is None:
        unpriced.append("dma_time(bytes counted)")
    issue = None
    bd_us = (op.get("host") or {}).get("bd_us", params.get("host_bd_us"))
    bd_us_src = "op" if "bd_us" in (op.get("host") or {}) else ("params" if bd_us is not None else None)
    if bd_us is None and params.get("bdgraph_bd_us") is not None:
        bd_us, bd_us_src = params["bdgraph_bd_us"], "board_fit"
    if f["bd_write_path"] is None:
        unpriced.append("bd_write_path")
    elif f["bd_configs_per_launch"] is None:
        unpriced.append("rounds_unknown")
    elif bd_us is None:
        unpriced.append("host_bd_us")
    elif bd_us_src == "board_fit" and params.get("bdgraph_fit_path") not in (None, f["bd_write_path"]):
        unpriced.append(f"issue(bd_write_path={f['bd_write_path']}; board fit is {params['bdgraph_fit_path']})")
    else:
        issue = f["bd_configs_per_launch"] * float(bd_us) / 1e3
        fit = params.get("bdgraph_issue_fit") if bd_us_src == "board_fit" else None
        p4j = bdgraph.issue_fit_ms(fit, f["bd_configs_per_launch"], floor, params) if fit else None
        if p4j is not None:
            issue, bd_us_src = p4j, "board_fit_p4j"
    runb = bdgraph.ddr_runb(floor, f["ddr_bytes"], params.get("bdgraph_run_b_factor"), issue)
    if f["locked_bds"]:
        unpriced.append("locks(AM020)")
    if f["runtime_bds_skipped"]:
        unpriced.append(f"runtime_bds({f['runtime_bds_skipped']})")
    return {"kernel": op.get("kernel"), "recipe": op.get("recipe"), "wait": None, "compute": None,
            "ddr": floor["ms"] if floor else None, "ddr_floor": floor, "ddr_runb": runb, "l_ms": None, "issue_ms": issue, "bd_us": bd_us if issue is not None else None,
            "bd_us_source": bd_us_src if issue is not None else None, "bound": None, "rounds": f["rounds"], "bdgraph": f,
            "bytes_in": op.get("bytes_in"), "bytes_out": op.get("bytes_out"), "unpriced": unpriced}


def _sum(rows: List[Dict[str, Any]], key: str) -> Optional[float]:
    v = [r[key] for r in rows if r.get(key) is not None]
    return sum(v) if v else None


def estimate(group: Any, board: str = "palmyra213", params: Optional[Dict[str, Any]] = None,
             cycles: Optional[str] = None) -> Dict[str, Any]:
    """Predicted ms of one execution of ``group`` (see module docstring for the schema)."""
    prm = params if params is not None else board_params(board)
    if "bdgraph_bd_us" not in prm:
        from .bdgraph import BD_US_FIT, BD_US_FIT_PATH, ISSUE_FIT
        if board in BD_US_FIT:
            prm = {**prm, "bdgraph_bd_us": BD_US_FIT[board]}
            if board in BD_US_FIT_PATH and "bdgraph_fit_path" not in prm:
                prm = {**prm, "bdgraph_fit_path": BD_US_FIT_PATH[board]}
        if board in ISSUE_FIT and "bdgraph_issue_fit" not in prm:
            prm = {**prm, "bdgraph_issue_fit": ISSUE_FIT[board]}
    if "bdgraph_run_b_factor" not in prm:
        from .bdgraph import RUN_B_FACTOR_FIT
        if board in RUN_B_FACTOR_FIT:
            prm = {**prm, "bdgraph_run_b_factor": RUN_B_FACTOR_FIT[board]}
    table = load_cycles(cycles)
    ops = _ops(group)
    launches = int(group.get("launches", 1)) if isinstance(group, dict) else 1
    rows = [estimate_op(op, prm, table) for op in ops]
    wait, l_ms, issue = _sum(rows, "wait"), _sum(rows, "l_ms"), _sum(rows, "issue_ms")
    comp, ddr = _sum(rows, "compute"), _sum(rows, "ddr")
    unpriced = sorted({u for r in rows for u in r["unpriced"]})
    bounds = {r["bound"] for r in rows if r["bound"] is not None}
    x = (lambda v: v * launches if v is not None else None)
    return {"kload": None, "cfg": None, "start": None, "glue": None,
            "wait": x(wait), "compute": x(comp), "ddr": x(ddr), "l_ms": x(l_ms), "issue_ms": x(issue),
            "total_ms": ((wait or 0.0) + (l_ms or 0.0) + (issue or 0.0)) * launches,
            "bound": bounds.pop() if len(bounds) == 1 else ("mixed" if bounds else None), "launches": launches,
            "board": board, "unpriced": unpriced, "ops": rows}
