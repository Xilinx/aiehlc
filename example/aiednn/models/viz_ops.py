###############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
"""ops.html: the operators aiednn supports, where each runs, and which other DNN libraries ship them.

Nothing is hand-copied into this script, so the page follows the tree:
  * the comparison table is the parity scoreboard in doc/design/aiednn.md (the section headed
    "Parity scoreboard"): its first column names the libraries that ship the operator, the next the AIE
    implementations for inference (the scoreboard's training and board-time columns and its backward-pass
    clauses are left out);
  * the inventory is read from the sources: every create function in include/aiednn.h, the AIE
    kernel entries of the kernel library (aiegraph.kernel_lib over src/aietensorop, with their
    AIEK_SPEC boards), and every ref: (host CPU) implementation name in src/aiednn.

    PYTHONPATH=src python3 example/aiednn/models/viz_ops.py    # writes worklocal/aiednn/models/ops.html
"""

from __future__ import annotations

import html
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "worklocal" / "aiednn" / "models" / "ops.html"
DOC = REPO / "doc" / "design" / "aiednn.md"


def _md_table(text: str, heading: str) -> tuple:
    """(section title, header cells, rows) of the first markdown table under the ## heading containing *heading*"""
    i = next(k for k, ln in enumerate(text.splitlines()) if ln.startswith("## ") and heading in ln)
    lines = text.splitlines()[i:]
    title = lines[0][3:].strip()
    rows = []
    for ln in lines[1:]:
        if ln.startswith("## "):
            break
        if ln.startswith("|"):
            rows.append([c.strip() for c in _split_row(ln)])
        elif rows:
            break
    return title, rows[0], [r for r in rows[2:]]


def _split_row(ln: str) -> list:
    """cells of a markdown table row; a '|' inside `code` is part of the cell"""
    cells, cur, code = [], "", False
    for ch in ln.strip()[1:-1]:
        if ch == "`":
            code = not code
        if ch == "|" and not code:
            cells.append(cur)
            cur = ""
        else:
            cur += ch
    cells.append(cur)
    return cells


def _inline(md: str) -> str:
    """the markdown the scoreboard uses -> html: `code`, **bold**"""
    s = html.escape(md)
    s = re.sub(r"`([^`]+)`", r"<code>\1</code>", s)
    return re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", s)


def _libs(cell: str, names: list) -> dict:
    """which of *names* ship the operator: named in the operator cell, or '(all ...)'"""
    head = cell.split("(", 1)
    paren = head[1] if len(head) > 1 else ""
    if re.match(r"\s*all\b", paren):
        return {n: True for n in names}
    return {n: n.lower() in cell.lower() for n in names}


def _where(cell: str) -> str:
    """aie | partial | host | none for an inference cell: the scoreboard bolds what stays on ref:"""
    c = cell.strip()
    if c in ("--", "—", "-"):
        return "none"
    if c.startswith("**ref:") or c.startswith("**ref"):
        return "host"
    if "ref:**" in c:
        return "partial"
    return "aie"


def _no_backward(cell: str) -> str:
    """the cell without its backward-pass / training clauses (split at ';' and top-level ',')"""
    drop = re.compile(r"\b(bwd|backward|training|train|BPTT|dK|dV)\b")
    keep = [c for c in re.split(r";\s*", cell) if not drop.search(c)]
    keep = [", ".join(p for p in re.split(r",\s*(?![^()]*\))", c) if not drop.search(p)) for c in keep]
    return "; ".join(k for k in keep if k.strip()) or "--"


def _creates() -> list:
    text = (REPO / "include" / "aiednn.h").read_text()
    names = set(re.findall(r"\b(aiednn_[a-z0-9_]*_create[a-z0-9_]*)\s*\(", text))
    return sorted(n for n in names if not re.search(r"_backward|_training|dropout|ctc_loss", n))   # inference only


def _ref_impls() -> list:
    """ref: implementations of inference operators"""
    names = set()
    for p in (REPO / "src" / "aiednn").glob("*"):
        if p.suffix in (".cc", ".h"):
            names |= set(re.findall(r'"(ref:[a-z0-9_]+)"', p.read_text(errors="replace")))
    return sorted(n for n in names if not re.search(r"dropout|ctc_loss", n))


def _aie_entries() -> list:
    import aiegraph as ag
    out = []
    for e in ag.kernel_lib(REPO / "src" / "aietensorop", quiet=True):
        sp = e.contract.spec
        out.append(dict(name=e.name, entry=sp.entry, family=e.contract.op, boards=list(sp.targets),
                        source=e.contract.source.name))
    return sorted(out, key=lambda d: (d["source"], d["name"]))


CSS = """body{font:14px system-ui,sans-serif;margin:24px;max-width:1500px}table{border-collapse:collapse;margin:8px 0 28px}
td,th{padding:6px 9px;border-bottom:1px solid #ddd;text-align:left;vertical-align:top}th{background:#f4f4f2;position:sticky;top:0}
td.c{text-align:center;font-size:16px}code{font-size:12px;background:#f3f3f1;padding:0 3px;border-radius:3px}
.aie{background:#fbe3d8}.partial{background:#fdf1ea}.host{background:#dce9f9}.none{color:#999}
.pill{display:inline-block;padding:1px 8px;border-radius:10px;font-size:12px;font-weight:600}
.pill.aie{background:#eb6834;color:#fff}.pill.partial{background:#f4a97f;color:#fff}.pill.host{background:#2a78d6;color:#fff}
.pill.none{background:#e6e6e3;color:#777}.yes{color:#1b8a5a}.no{color:#bbb}
.small{font-size:12px;color:#555}.cols{columns:3;font-size:12.5px}"""

PILL = {"aie": "AIE", "partial": "AIE, part on host", "host": "host CPU only", "none": "n/a"}


def main() -> Path:
    text = DOC.read_text()
    _, _, rows = _md_table(text, "Parity scoreboard")
    libs = [n for n in ("oneDNN", "cuDNN", "hipDNN")]
    recs = []
    for r in rows:
        if len(r) < 4:
            continue
        # inference only: training-only operators (dropout, CTC loss) and "training" attributions leave the label
        op = re.sub(r"\s*training\)", ")", re.sub(r"(Dropout, |, CTC loss)", "", r[0]))
        inf = re.sub(r"\s*\(dropout [\d.]+, grid_sample [\d.]+\)", "", _no_backward(r[1]))
        recs.append(dict(op=op, libs=_libs(op, libs), inf=inf, w_inf=_where(inf)))
    creates, refs, aie = _creates(), _ref_impls(), _aie_entries()

    trs = []
    for x in recs:
        lib_cells = "".join(f'<td class="c {"yes" if x["libs"][n] else "no"}">{"&#10003;" if x["libs"][n] else "&ndash;"}</td>'
                            for n in libs)
        ext = "" if any(x["libs"].values()) else ' <span class="pill none">aiednn fusion / extension</span>'
        trs.append(f'<tr><td>{_inline(x["op"])}{ext}</td>{lib_cells}'
                   f'<td class="{x["w_inf"]}"><span class="pill {x["w_inf"]}">{PILL[x["w_inf"]]}</span><br>{_inline(x["inf"])}</td></tr>')

    by_src = {}
    for e in aie:
        by_src.setdefault(e["source"], []).append(e)
    aie_rows = "".join(f'<tr><td><code>{html.escape(e["name"])}</code></td><td><code>{html.escape(e["entry"])}</code></td>'
                       f'<td>{html.escape(e["family"])}</td><td>{", ".join(b.upper() for b in e["boards"])}</td>'
                       f'<td><code>{html.escape(e["source"])}</code></td></tr>' for e in aie)

    page = f"""<!doctype html><meta charset=utf-8><title>aiednn operators vs other DNN libraries</title><style>{CSS}</style>
<h1>aiednn operators</h1>
<p><a href="index.html">&larr; model pages</a></p>
<p class=small><span class="pill aie">AIE</span> runs on the AIE array<br>
<span class="pill partial">AIE, part on host</span> some variants run on the host<br>
<span class="pill host">host CPU only</span> no AIE path<br>
<span class="pill none">aiednn fusion / extension</span> no other library ships it</p>
<h2>Operator families (inference): aiednn vs oneDNN / cuDNN / hipDNN</h2>
<table><tr><th>operator family</th>{"".join(f"<th>{n}</th>" for n in libs)}<th>in aiednn</th></tr>
{chr(10).join(trs)}</table>
<h2>AIE kernel entries ({len(aie)})</h2>
<p class=small>Each is a mode or entry of an aiehlc kernel source; dispatch reaches them directly or through the
<code>blocked:</code> / <code>im2col+</code> / <code>gemm+</code> composites named in the table above.</p>
<table><tr><th>kernel</th><th>entry</th><th>family</th><th>boards (AIEK_SPEC target)</th><th>source</th></tr>{aie_rows}</table>
<h2>Create functions in <code>include/aiednn.h</code> ({len(creates)})</h2>
<div class=cols>{"<br>".join(f"<code>{c}</code>" for c in creates)}</div>
<h2>Host (<code>ref:</code>) implementations ({len(refs)})</h2>
<div class=cols>{"<br>".join(f"<code>{r}</code>" for r in refs)}</div>
"""
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(page)
    return OUT


if __name__ == "__main__":
    print(main())
