###############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
"""index.html for the model-flow pages in worklocal/aiednn/models (run_models.py --html).

One table per board: every page's embedded data (aiegraph.dnn_viz) gives the model, its board
run (MODEL line of the run.log the page was built against) and its megakernel use. A second
VEK385 table lists the graph-path (torch.compile / PT2E -> partition graph API) board runs
from worklocal/aiednn/torch_compile_board, which have no per-op page.

    python3 example/aiednn/models/viz_index.py          # writes worklocal/aiednn/models/index.html
"""

from __future__ import annotations

import datetime as _dt
import html
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "worklocal" / "aiednn" / "models"
GRAPH = REPO / "worklocal" / "aiednn" / "torch_compile_board"
BOARD_NAME = {"vek280": "VEK280", "vek385": "VEK385"}
BENCH = ("resnet50", "vit_b_16", "bert_base", "gpt2_small", "tiny_llama", "yolov8n")

_MODEL = re.compile(r"^MODEL (\S+) .*?time_ms=([\d.]+) relerr=(\S+) top1=(-?\d+) ref_top1=(-?\d+)", re.M)


def _page_data(p: Path) -> dict | None:
    s = p.read_text(errors="replace")
    m = re.search(r'<script type="application/json" id="data">(.*?)</script>', s, re.S)
    return json.loads(m.group(1).replace("<\\/", "</")) if m else None


def _row(p: Path, d: dict) -> tuple:
    """(board key, sort key, html row) of one model page"""
    S, b = d["summary"], d.get("board") or {}
    mk = d.get("megakernel") or dict(modes=[], build={})     # pages written before the megakernel view
    tgt = (d.get("target") or {}).get("key") or ("vek385" if "/vek385/" in b.get("path", "") else "vek280")
    tail = p.stem[len(d["model"]):]            # the options after the model name (efficientnet_b0 is not batch 0)
    mesh = re.search(r"_(\d+x\d+)", tail)
    batch = re.search(r"_b(\d+)(?:_|$)", tail)
    used = [m["mode"] for m in mk["modes"] if m["ops"]]
    built = mk.get("build") or {}
    comp = ", ".join(k for k, v in (built.get("modes") or {}).items() if v) or ("all" if not built else "GEMM only")
    pm = f'{built["pm_bytes"] / 1024:.1f} KB' if built.get("pm_bytes") else "-"
    bm = b.get("model") or {}
    bt = f'{bm["time_ms"]:.1f} ms' if bm.get("time_ms") else "-"
    ok = "-" if not bm else ("PASS" if b.get("passed") or (bm.get("top1") == bm.get("ref_top1") and not bm.get("gemm_bad"))
                             else "FAIL")
    exact = f'{bm["gemm_checked"] - bm["gemm_bad"]}/{bm["gemm_checked"]}' if bm.get("gemm_checked") else "-"
    when = _dt.datetime.fromtimestamp((REPO / b["path"]).stat().st_mtime).strftime("%Y-%m-%d %H:%M") \
        if b.get("path") and (REPO / b["path"]).is_file() else "-"
    flags = [f for f in ("fastest",) if f"_{f}" in p.stem]
    cells = [f'<a href="{html.escape(p.name)}">{html.escape(d["model"])}</a>', d["mode"],
             mesh.group(1) if mesh else "4x4", batch.group(1) if batch else "1", ", ".join(flags) or "aie",
             bt, str(S["ops"]), str(S["aie_ops"]), str(S.get("counted_launches") or "-"), ", ".join(used) or "-",
             comp, pm, f'{bm.get("top1", "-")} ({bm.get("ref_top1", "-")})', exact, ok, when]
    num = {5, 6, 7, 8, 11, 12, 13}
    tr = "<tr>" + "".join(f'<td{" class=n" if i in num else ""}>{c}</td>' for i, c in enumerate(cells)) + "</tr>"
    order = (d["model"] not in BENCH, BENCH.index(d["model"]) if d["model"] in BENCH else 0, d["model"],
             -(p.stat().st_mtime))
    return tgt, order, tr


def _graph_rows() -> list:
    rows = []
    for log in sorted(GRAPH.glob("*/vek385/run.log")):
        name = log.parent.parent.name
        text = log.read_text(errors="replace").replace("\r", "")
        m = _MODEL.search(text)
        if not m:
            continue
        model = re.sub(r"_pt2e.*", "", name)
        fuse = re.search(r"_fuse(\d)", name)
        parts = re.search(r"partitions=(\d+)", text)
        sw = re.search(r"MEGAKERNEL switches=(\d+)", text)
        guard = re.search(r"aiednn_switch_guard: (\d+) primitives reloaded a core ELF, (\d+) moved", text)
        verdict = "FAIL" if re.search(r"^FAIL", text, re.M) else "PASS"
        ref2 = re.search(r"^REF2 (\S+) relerr=(\S+) top1=(-?\d+)", text, re.M)
        when = _dt.datetime.fromtimestamp(log.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
        cells = [html.escape(model), fuse.group(1) if fuse else "1", "fastest" if "_fastest" in name else "aie",
                 f"{float(m.group(2)):.1f} ms", parts.group(1) if parts else "-", sw.group(1) if sw else "-",
                 f"{guard.group(2)} of {guard.group(1)}" if guard else "-", f"{float(m.group(3)):.1e}",
                 f"{m.group(4)} ({m.group(5)})",
                 f"{ref2.group(1)} {float(ref2.group(2)):.1e}, top-1 {ref2.group(3)}" if ref2 else "-", verdict, when]
        order = (model not in BENCH, BENCH.index(model) if model in BENCH else 0, model)
        rows.append((order, "<tr>" + "".join(f'<td{" class=n" if i in (3, 4, 5, 7) else ""}>{c}</td>'
                                            for i, c in enumerate(cells)) + "</tr>"))
    return [r for _, r in sorted(rows)]


HEAD = ("<tr><th>model</th><th>mode</th><th>mesh</th><th>batch</th><th>preference</th><th>board time</th>"
        "<th>primitives</th><th>on AIE</th><th>AIE launches</th><th>megakernel modes used</th>"
        "<th>modes compiled in its board build</th><th>core program</th><th>top-1 (ref)</th>"
        "<th>AIE layers exact</th><th>board run</th><th>run at</th></tr>")
GHEAD = ("<tr><th>model</th><th>fusion level</th><th>preference</th><th>board time</th><th>partitions</th>"
         "<th>megakernel switches</th><th>switch guard: moved to ref:</th><th>rel err vs golden</th>"
         "<th>top-1 (golden)</th><th>PT2E eager (reported)</th><th>result</th><th>run at</th></tr>")


def main() -> Path:
    by_board = {}
    for p in sorted(OUT.glob("*.html")):
        if p.name == "index.html":
            continue
        d = _page_data(p)
        if d is None:
            continue
        tgt, order, tr = _row(p, d)
        by_board.setdefault(tgt, []).append((order, tr))
    now = _dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    parts = ["""<!doctype html><meta charset=utf-8><title>aiednn models on VEK280 / VEK385</title>
<style>body{font:14px system-ui,sans-serif;margin:24px}table{border-collapse:collapse;margin-bottom:28px}
td,th{padding:5px 10px;border-bottom:1px solid #ddd;text-align:left}td.n{text-align:right}a{color:#2a78d6}</style>
<h1>aiednn models on the board</h1>""",
             f"<p>Click a model for its flow page (ops, buffers, megakernel modes, the AIE array). Board time is the "
             f"board's <code>MODEL ... time_ms</code>; per-op figures on each page are workstation emulation. "
             f"Benchmark models first, then the rest; within a model, newest first. Generated {now}. "
             f'<a href="ops.html">Operators aiednn supports, compared with oneDNN / cuDNN / hipDNN &rarr;</a></p>']
    for key in ("vek385", "vek280"):
        rows = [tr for _, tr in sorted(by_board.get(key, []))]
        if not rows:
            continue
        host = "Arm Cortex-A78" if key == "vek385" else "Arm Cortex-A72"
        parts.append(f"<h2>{BOARD_NAME[key]} -- direct path (<code>run_models.py --mlir</code>), host {host}</h2>")
        parts.append("<table>" + HEAD + "\n" + "\n".join(rows) + "</table>")
        if key == "vek385":
            g = _graph_rows()
            if g:
                parts.append("<h2>VEK385 -- graph path (<code>torch_compile_board.py</code>: torch.compile / PT2E int8 "
                             "-&gt; partition graph API)</h2><p>No per-op page: the graph path's ops are partitions "
                             "the library forms at run time. The golden is aiednn's workstation-emulation run of the "
                             "same partitions where the column PT2E eager is filled (earlier runs checked against "
                             "PT2E eager itself).</p>")
                parts.append("<table>" + GHEAD + "\n" + "\n".join(g) + "</table>")
    out = OUT / "index.html"
    out.write_text("\n".join(parts) + "\n")
    return out


if __name__ == "__main__":
    print(main())
