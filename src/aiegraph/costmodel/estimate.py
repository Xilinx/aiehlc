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
                         "bd_write_path": "jit_mmio|jit_ctrlpkt|aot_window|uart_degraded",  # required to price issue_ms;
                                                      #   the board fit only prices its own path (palmyra213: jit_ctrlpkt)
                         "trip_count": int}}          # optional; else start_io.loop_range (see bdgraph.py)
                                                      # -> counts only + issue_ms + ddr floor (report only,
                                                      #    bdgraph.ddr_floor); K terms listed in "unpriced"

Site mode (M6.1): ``estimate({"unet_build": dir, "cfg": "off"|"aot-kco", "sites": None|[...]}, board="palmyra213",
params="m3122")`` returns the frozen board-validated K+L+N model per UNet site (see estimate_sites).

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
    if f["plain_shim_bds_large"] and f["bd_write_path"] == "aot_window":
        # such BDs are written directly, outside the AOT window (Amit des-json-gap section 1)
        unpriced.append(f"plain_shim_bds_large({f['plain_shim_bds_large']}) outside aot_window")
    return {"kernel": op.get("kernel"), "recipe": op.get("recipe"), "wait": None, "compute": None,
            "ddr": floor["ms"] if floor else None, "ddr_floor": floor, "ddr_runb": runb, "l_ms": None, "issue_ms": issue, "bd_us": bd_us if issue is not None else None,
            "bd_us_source": bd_us_src if issue is not None else None, "bound": None, "rounds": f["rounds"], "bdgraph": f,
            "bytes_in": op.get("bytes_in"), "bytes_out": op.get("bytes_out"), "unpriced": unpriced}


def _sum(rows: List[Dict[str, Any]], key: str) -> Optional[float]:
    v = [r[key] for r in rows if r.get(key) is not None]
    return sum(v) if v else None


# ------------------------------------------------------------------ M6.1 site mode: frozen K+L+N (M34 PZ)
# The board-validated model of script/perf/kln_frozen.sh with KLN_K_JSON=k m3122 (M34 PZ), run in-process on the
# build dir only.  Frozen parameter sets: {board: {name: {role: params-relative path}}}.
KLN_PARAMS = {"palmyra213": {"m3122": {
    "k": "k_palmyra213.m3122.json",            # K_hat static (k_diag --khat-static --params)
    "l": "l_palmyra213.m3121.json",            # L (m3_site --lpred --kfirst site --params)
    "n": "n_unet_palmyra213.m3121.json",       # N + ovl (n_fit --predict --ovl --refit-a no --params)
    "kcyc": "kcyc_unet_iss.json",              # core cycles (kcyc_unet --iss-unet)
    "kdw": "kdw_palmyra213.m3112c.json",       # dw_hat (k_dmafeat --dwpredict --dw-params)
}}}
# fixed (non-fitted) side inputs the chain also reads: n_fit --off-corr default (cfg=off in-window rt bdcfg/startio
# coefficients) and the kernel ISS table of k_core_check.ISS
KLN_AUX = {"off_corr": "l_palmyra213.json"}
KLN_CFGS = ("off", "aot-kco")


def _md5(path: Path) -> str:
    import hashlib
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def _kcyc_sites(build: str, iss: str, iss_unet: str) -> Dict[Any, Dict[str, Any]]:
    """kcyc_unet static core cycles per site {(kind, cid): rec} (= k_core_check.kcyc with --iss-unet explicit)"""
    import contextlib
    import io
    import os
    import tempfile
    import kcyc_unet
    with tempfile.TemporaryDirectory() as td:
        out = os.path.join(td, "kcyc.json")
        with contextlib.redirect_stdout(io.StringIO()):
            rc = kcyc_unet.main(["--build-root", build, "--iss", iss, "--iss-unet", iss_unet, "--out", out])
        doc = json.loads(Path(out).read_text()) if os.path.exists(out) else dict(sites=[])
    if rc != 0:
        raise RuntimeError("kcyc_unet rc=%d on %s" % (rc, build))
    return {(r["kind"], r["cid"]): r for r in doc["sites"]}


def estimate_sites(spec: Dict[str, Any], board: str = "palmyra213", params: str = "m3122",
                   params_dir: Optional[Path] = None) -> Dict[str, Any]:
    """Site mode: the frozen, board-validated K+L+N model (M34 PZ) for every launches>0 site of one UNet build cfg,
    from the build dir only (no board log).

        spec = {"unet_build": BUILD_ROOT (holding unet_<cfg>/) or the unet_<cfg> dir itself,
                "cfg": "off" | "aot-kco", "sites": None | ["conv0", "linear3", ...]}

    Per site (ms; totals over the site's launches unless marked per launch):
        L      = L_hat   (m3_site.l_sites + offset/kswitch terms of the L json)
        N      = N_hat   (n_fit.npred_site), N_ovl = beta * ovl_hat (the glue that overlaps the AIE window)
        core   = alpha * core_hat (kcyc_unet ISS cycles x launches), dma = gamma * dw_hat x launches (k_dmafeat)
        K      = K_hat x launches, K_hat = max(0, body - N_ovl/launches) + t0 + kfirst/launches,
                 body = core + dma (form sum) or max(core, dma) (form max), per launch
        N_pre  = N - N_ovl, N_post = (t0 x launches + kfirst) (fixed K terms outside the overlap)
        so  T_hat x launches = L + N_pre + max(busy, N_ovl) + N_post  when busy >= N_ovl (busy = body x launches),
        and T_hat = (K + L + N) / launches  (ms per launch) always -- kln_site.predict_launch.
    Every value goes through the same text rounding as the kln_frozen.sh TSV chain (L/N/ovl %.4f, core %.6f,
    dw %.6g) so the result equals the frozen model's T_hat bit for bit.  Nothing is fitted.
    Returns {"sites": [...], "total_ms", "params": {relpath: md5} (the 5 frozen files), "aux": {relpath: md5},
             "unpriced": [], "cfg", "board", "model"}."""
    import sys
    pd = Path(params_dir) if params_dir else PARAMS_DIR
    perf = str(pd.parent)
    if perf not in sys.path:
        sys.path.insert(0, perf)
    try:
        files = KLN_PARAMS[board][params]
    except KeyError:
        raise ValueError("no frozen K+L+N params %r for board %r" % (params, board))
    cfg = spec.get("cfg")
    if cfg not in KLN_CFGS:
        raise ValueError("cfg must be one of %s, got %r" % (KLN_CFGS, cfg))
    root = Path(spec["unet_build"])
    if root.name == "unet_" + cfg:
        root = root.parent
    build = str(root / ("unet_" + cfg))
    if not Path(build).is_dir():
        raise FileNotFoundError("no UNet build dir %s" % build)
    P = {k: pd / v for k, v in files.items()}
    rel = lambda p: str(Path(p).resolve().relative_to(pd.parents[2]))  # noqa: E731
    pmd5 = {rel(P[k]): _md5(P[k]) for k in ("k", "l", "n", "kcyc", "kdw")}
    aux = {rel(pd / v): _md5(pd / v) for v in KLN_AUX.values()}

    import k_core_check
    import k_diag
    import k_dmafeat
    import m3_site
    import n_fit
    r4 = lambda x: float("%.4f" % x)  # noqa: E731  (lpred / npred TSV)
    # static site list: launches>0 sites of n_feat.sites in graph_run order, part = the log's PROF part (lpart)
    srows = n_fit.static_rows(build)
    live = [s for s in srows if s["launches"] > 0]
    # L (m3_site --lpred --kfirst site)
    lprm = json.loads(P["l"].read_text())
    cp = lprm["cfg"].get(cfg, {})
    coef, terms = cp.get("coef", {}), bool(cp.get("fit_runs"))
    lh = m3_site.l_sites(build, cfg, str(P["l"]), "site")
    lf = m3_site.site_launch_feats(build) if terms else {}
    # N + ovl (n_fit --predict --ovl --refit-a no)
    npc = json.loads(P["n"].read_text())[cfg]
    if "ovl" not in npc:
        raise ValueError("N params %s cfg %s have no ovl term" % (P["n"], cfg))
    corr = [0.0] * len(live)
    if cfg == "off":
        cs = n_fit._off_corr(build, str(pd / KLN_AUX["off_corr"]))
        if [(k, c) for k, c, _ in cs] != [(s["kind"], s["cid"]) for s in live]:
            raise ValueError("off-corr sites vs n_feat sites differ")
        corr = [ms for _k, _c, ms in cs]
    # core (k_diag --kdelta core: kcyc_unet core_cyc x launches)
    kc = _kcyc_sites(build, k_core_check.ISS, str(P["kcyc"]))
    # dw (k_dmafeat --dwpredict): site_table on the static rows
    rows = [dict(s, part=s["lpart"]) for s in live]
    dwp = k_dmafeat.dw_load_params(str(P["kdw"]))[cfg]
    dws = {r[1]: float(r[6]) for r in k_dmafeat.dwp_rows(cfg, k_dmafeat.site_table_rows(cfg, rows, str(root)), dwp)}
    # K (k_diag --khat-static --params)
    kdoc = json.loads(P["k"].read_text())
    kcls, kform = kdoc["cfg"][cfg]["classes"], kdoc["cfg"][cfg]["form"]
    want = set(spec.get("sites") or [])
    out = []
    for j, s in enumerate(live):
        key, name, n = (s["kind"], s["cid"]), "%s%d" % (s["kind"], s["cid"]), int(s["launches"])
        if want and name not in want:
            continue
        L = r4(lh[key] + (m3_site.l_terms(coef, lf[key]) if terms else 0.0))
        nh, oh = n_fit.npred_site(npc, s, corr[j])
        N, ovl = r4(nh), r4(oh)
        kr = kc.get(key)
        if kr is None:
            raise KeyError("site %s: no kcyc_unet core cycles" % name)
        core_hat = float("%.6f" % (float(kr["core_cyc"]) * n / k_diag.AIE_MHZ / 1e3))
        st = dict(launches=n, core_hat_ms=core_hat, dw_hat_ms=dws[str(s["lpart"])], ovl_hat=ovl)
        cls = k_diag.khats_cls(k_diag.site_class(s["kind"], s.get("kernels") or []))
        pk = {k: float(kcls[cls][k]) for k in k_diag.KHATS_PNAMES}
        fc = k_diag.khats_cls_form(kcls[cls], kform)
        f = k_diag.khats_feat(st)
        kh = k_diag.khats_predict(pk, fc, f)
        K = kh * n
        core, dma, novl = pk["alpha"] * f["core_l"] * n, pk["gamma"] * f["dw_l"] * n, pk["beta"] * f["ovl_l"] * n
        out.append(dict(site=name, part=s["lpart"], cls=cls, form=fc, launches=n, L=L, N=N,
                        N_pre=N - novl, N_ovl=novl, N_post=pk["t0"] * n + pk["kfirst"],
                        core=core, dma=dma, K=K, T_hat=(K + L + N) / n))
    if want - {r["site"] for r in out}:
        raise KeyError("unknown sites %s for cfg %s" % (sorted(want - {r["site"] for r in out}), cfg))
    return {"sites": out, "total_ms": sum(r["T_hat"] * r["launches"] for r in out), "params": pmd5, "aux": aux,
            "unpriced": [], "cfg": cfg, "board": board,
            "model": "K+L+N frozen (M34 PZ): T_hat = (K_hat x launches + L_hat + N_hat) / launches"}


def estimate(group: Any, board: str = "palmyra213", params: Any = None,
             cycles: Optional[str] = None) -> Dict[str, Any]:
    """Predicted ms of one execution of ``group`` (see module docstring for the schema).
    ``group = {"unet_build": dir, "cfg": ..., "sites": ...}`` selects site mode (estimate_sites; params = name of
    the frozen set, default "m3122")."""
    if isinstance(group, dict) and "unet_build" in group:
        return estimate_sites(group, board, params if isinstance(params, str) else "m3122")
    if isinstance(params, str):
        raise ValueError("params name %r is only valid in site mode (group with 'unet_build')" % params)
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
            "board": board, "unpriced": unpriced, "ops": rows,
            # hard HW/runtime limits broken by a bdgraph op: the design will not run as built, total_ms is moot
            "hw_violations": [v for r in rows for v in (r.get("bdgraph") or {}).get("hw_violations", [])]}
