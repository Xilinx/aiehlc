"""``python -m aiegraph.costmodel <subcommand>`` -- one command line over the existing cost-model code.

    estimate  --net unet|resnet18|yolo26n --build ROOT --cfg CFG [--sites a,b] [--params NAME]   (net/site mode)
    estimate  --kernel NAME [--cycles JSON] [--bd-us X] [--launches N]                          (one kacc kernel)
    estimate  --group-json FILE                                     (any estimate() input, see estimate.py:1-40)
    fit       k|l|n|mb|scmb|kcyc|kcore|knet|kdiag  [tool args ...]  (runs the fitting tool, python -I)
    report    scatter|params [tool args ...]                        (report_scatter.py / params_table.py)
    tool      NAME [args ...]                                       (any tool in costmodel/tools/)
    tools                                                           (list the tools)
    params                                                          (frozen params files + md5)
    selftest                                                        (offline: every tool --selftest + frozen md5s)

Nothing here re-implements the model: estimate goes to estimate.estimate(), the rest runs the tool scripts exactly
the way the board gates do (``python3 -I <tool>.py``), so their output and rc are unchanged.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

from . import tools as _tools
from .estimate import PARAMS_DIR, REPO_ROOT, estimate

FIT = {"k": "costmodel_fit", "l": "l_fit", "n": "n_fit", "mb": "mb_fit", "scmb": "scmb_fit", "kcyc": "kcyc_unet",
       "kcore": "kcore_net", "knet": "k_net", "kdiag": "k_diag"}
REPORT = {"scatter": "report_scatter", "params": "params_table"}
# tools with an offline --selftest (no board, no build dir) and the line each prints on success
SELFTESTS = ("k_noise", "k_dmafeat", "k_ovl_check", "kln_site", "k_diag", "t_noise", "net_sites", "tsv_cmp",
             "net_kdiag")


def run_tool(name: str, args: List[str]) -> int:
    """run tools/<name>.py under ``python -I`` (as the gates do); output passes straight through"""
    return subprocess.call([sys.executable, "-I", str(_tools.path(name))] + list(args))


def _md5(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


def _print_net(r: dict) -> None:
    print("ESTIMATE net=%s cfg=%s board=%s sites=%d total_ms=%.4f" % (r.get("net", "unet"), r["cfg"], r["board"],
                                                                     len(r["sites"]), r["total_ms"]))
    print("model: %s" % r["model"])
    for f, m in sorted(r["params"].items()):
        print("params %s md5=%s" % (f, m))
    for s in r["sites"]:
        print("site=%-10s launches=%-5d T_hat_ms=%.4f site_ms=%.4f" % (s["site"], s["launches"], s["T_hat"],
                                                                     s["T_hat"] * s["launches"]))


def _print_group(r: dict) -> None:
    f = lambda v: "-" if v is None else "%.4f" % v  # noqa: E731
    print("ESTIMATE board=%s launches=%d total_ms=%s wait=%s compute=%s ddr=%s l_ms=%s issue_ms=%s bound=%s"
          % (r["board"], r["launches"], f(r["total_ms"]), f(r["wait"]), f(r["compute"]), f(r["ddr"]),
             f(r["l_ms"]), f(r.get("issue_ms")), r["bound"]))
    print("unpriced (counted as 0 in total_ms): %s" % (", ".join(r["unpriced"]) or "-"))


def cmd_estimate(a) -> int:
    if a.net:
        sites = a.sites.split(",") if a.sites else None
        r = estimate({"net": a.net, "build": a.build, "cfg": a.cfg, "sites": sites}, board=a.board, params=a.params)
    elif a.kernel or a.group_json:
        if a.group_json:
            group = json.loads(Path(a.group_json).read_text())
        else:
            op = {"kernel": a.kernel, "recipe": a.recipe}
            if a.bd_us is not None:
                op["host"] = {"bd_us": a.bd_us}
            group = {"ops": [op], "launches": a.launches}
        r = estimate(group, board=a.board, cycles=a.cycles)
    else:
        print("estimate: give --net (with --build/--cfg), --kernel or --group-json", file=sys.stderr)
        return 2
    if a.json:
        print(json.dumps(r, indent=1, sort_keys=True, default=str))
    elif "sites" in r:
        _print_net(r)
    else:
        _print_group(r)
    return 0


def cmd_params(_a) -> int:
    for p in sorted(PARAMS_DIR.iterdir()):
        print("%s  %s" % (_md5(p), p.relative_to(REPO_ROOT)))
    return 0


def check_frozen() -> List[str]:
    """every 'md5  relpath' line of params/*.md5 still matches the file -> list of errors"""
    err = []
    for mf in sorted(PARAMS_DIR.glob("*.md5")):
        for ln in mf.read_text().splitlines():
            if ln.strip():
                m, rp = ln.split(None, 1)
                f = REPO_ROOT / rp.strip()
                if not f.is_file() or _md5(f) != m:
                    err.append("%s: %s" % (mf.name, rp.strip()))
    return err


def cmd_selftest(_a) -> int:
    bad = []
    for t in SELFTESTS:
        out = subprocess.run([sys.executable, "-I", str(_tools.path(t)), "--selftest"], capture_output=True,
                             text=True)
        last = (out.stdout.strip().splitlines() or [""])[-1]
        print("COSTMODEL SELFTEST tool=%s rc=%d %s" % (t, out.returncode, last[:120]))
        if out.returncode != 0:
            bad.append(t)
    err = check_frozen()
    print("COSTMODEL SELFTEST frozen_md5 files=%d bad=%d %s" % (len(list(PARAMS_DIR.glob("*.md5"))), len(err),
                                                               "; ".join(err)))
    ok = not bad and not err
    print("COSTMODEL SELFTEST tools=%d bad=%d %s" % (len(SELFTESTS), len(bad) + len(err), "PASS" if ok else "FAIL"))
    return 0 if ok else 1


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m aiegraph.costmodel", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("estimate", help="predict ms with the frozen per-board params (estimate.estimate)")
    e.add_argument("--board", default="palmyra213")
    e.add_argument("--net", choices=("unet", "resnet18", "yolo26n"))
    e.add_argument("--build", help="build root holding <net>_<cfg>/")
    e.add_argument("--cfg", help="off | aot-kco | aot (yolo26n)")
    e.add_argument("--sites", help="comma list: unet conv0,linear3..; resnet18/yolo26n part1,part3..")
    e.add_argument("--params", help="frozen set name (unet m3122, resnet18 m41, yolo26n m42)")
    e.add_argument("--kernel", help="kernel name in the ISS cycles table, e.g. u_lin_1024x320x320")
    e.add_argument("--recipe", default="kacc_gemm_bf16")
    e.add_argument("--cycles", help="aie2ps_kernel_cycles.json (default estimate.CYCLES)")
    e.add_argument("--bd-us", type=float, help="host BD-issue us per BD (else params / default 5.0)")
    e.add_argument("--launches", type=int, default=1)
    e.add_argument("--group-json", help="file with an estimate() group (dict with ops, or a list of ops)")
    e.add_argument("--json", action="store_true", help="print the full estimate() dict")
    for name, table in (("fit", FIT), ("report", REPORT)):
        p = sub.add_parser(name, help="run a %s tool: %s" % (name, ", ".join("%s=%s" % kv for kv in table.items())))
        p.add_argument("which", choices=sorted(table))
        p.add_argument("args", nargs=argparse.REMAINDER)
    t = sub.add_parser("tool", help="run any tool in costmodel/tools by name")
    t.add_argument("name")
    t.add_argument("args", nargs=argparse.REMAINDER)
    sub.add_parser("tools", help="list the tools")
    sub.add_parser("params", help="list the frozen params files with md5")
    sub.add_parser("selftest", help="offline self tests of the tools + frozen md5 check")
    return ap


def main(argv: Optional[List[str]] = None) -> int:
    a = parser().parse_args(argv)
    if a.cmd == "estimate":
        return cmd_estimate(a)
    if a.cmd == "fit":
        return run_tool(FIT[a.which], a.args)
    if a.cmd == "report":
        return run_tool(REPORT[a.which], a.args)
    if a.cmd == "tool":
        return run_tool(a.name, a.args)
    if a.cmd == "tools":
        print("\n".join(_tools.names()))
        return 0
    if a.cmd == "params":
        return cmd_params(a)
    return cmd_selftest(a)
