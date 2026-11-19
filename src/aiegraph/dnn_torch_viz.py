"""The op graph at every stage of ``torch.compile(model, backend="aiednn")``.

``aiegraph.backends.aiednn.dump = <dir>`` makes ``dnn_torch_backend`` record each stage as it happens
and write ``<dir>/index.md`` -- one Mermaid chart per stage -- plus a ``.dot`` per stage and, when
graphviz is installed, its ``.svg``. Nothing is re-derived here: a stage is a snapshot of the graph the
backend is holding at that moment, so the charts show what really ran.

    1 dynamo        the fx graph Dynamo captured
    2 frozen        parameters Dynamo lifted to inputs -> constants (_freeze_params)
    3 conv+bn       BN folded into its conv (torch.fx.experimental.optimization.fuse)
    4 rewrites      _sink_permute, _inline_mha, _split_to_narrow, _select_to_narrow
    5 graph ops     each fx node as one op of the graph API; unmapped ones are wildcards
    6 partitions    the library's partitions: an anchor and the ops it fused, supported or not
    7 executed      the GraphModule torch runs: one node per compiled partition, the rest torch

Constants are left out unless ``consts`` is true; a consumer's label counts them instead.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

#: stage kind -> (mermaid class, dot fillcolor, dot fontcolor)
_STYLE = {
    "in":     ("sIn",    "#e8e8e8", "#000000"),
    "out":    ("sOut",   "#e8e8e8", "#000000"),
    "op":     ("sOp",    "#dbe9f6", "#000000"),
    "wild":   ("sWild",  "#f6e0db", "#000000"),
    "const":  ("sConst", "#f2f2f2", "#555555"),
    "aie":    ("sAie",   "#cfe9cf", "#000000"),
    "no":     ("sNo",    "#f6e0db", "#000000"),
    "torch":  ("sTorch", "#ededed", "#000000"),
}
_CLASSDEF = {
    "sIn":    "fill:#e8e8e8,stroke:#999,color:#000",
    "sOut":   "fill:#e8e8e8,stroke:#999,color:#000",
    "sOp":    "fill:#dbe9f6,stroke:#5b8db8,color:#000",
    "sWild":  "fill:#f6e0db,stroke:#c08070,color:#000",
    "sConst": "fill:#f2f2f2,stroke:#bbb,color:#555",
    "sAie":   "fill:#cfe9cf,stroke:#4f9a4f,color:#000",
    "sNo":    "fill:#f6e0db,stroke:#c08070,color:#000",
    "sTorch": "fill:#ededed,stroke:#aaa,color:#000",
}


def _esc(s: str) -> str:
    """a mermaid label: the characters that end a node shape, as entities"""
    for a, b in (("&", "#38;"), ('"', "#quot;"), ("[", "#91;"), ("]", "#93;"),
                 ("(", "#40;"), (")", "#41;"), ("{", "#123;"), ("}", "#125;")):
        s = s.replace(a, b)
    return s


def _id(name: str) -> str:
    return "n_" + "".join(c if c.isalnum() or c == "_" else "_" for c in str(name))


def _tmeta(node):
    """(shape, dtype) of an fx node's value, from whichever meta the stage has"""
    m = node.meta.get("tensor_meta")
    if m is None:
        m = node.meta.get("example_value", node.meta.get("val"))
    if m is None or not hasattr(m, "shape") or not hasattr(m, "dtype"):
        return None
    try:
        return tuple(int(d) for d in m.shape), str(m.dtype).replace("torch.", "")
    except (TypeError, ValueError):
        return None


def _shape_str(node) -> str:
    t = _tmeta(node)
    return "" if t is None else f"{t[1]}[{','.join(str(d) for d in t[0])}]"


class _Stage:
    def __init__(self, name, note):
        self.name, self.note = name, note
        self.nodes = []            # (id, label, sublabel, kind)
        self.edges = []            # (src id, dst id, label)
        self.groups = []           # (title, [ids], kind)


class Trace:
    """the stages, recorded as the backend walks them; ``write`` renders them"""

    def __init__(self, consts: bool = False):
        self.stages, self.consts = [], consts

    # ── recording ────────────────────────────────────────────────────────
    def fx(self, name: str, gm, note: str = "") -> None:
        """one fx GraphModule: call nodes, their shapes, the edges between them"""
        st = _Stage(name, note)
        mods = dict(gm.named_modules())
        nconst = {}
        for node in gm.graph.nodes:
            if node.op == "get_attr" or (node.op == "placeholder" and not node.users):
                for u in node.users:
                    nconst[u.name] = nconst.get(u.name, 0) + 1
                if not self.consts:
                    continue
        for node in gm.graph.nodes:
            if node.op == "placeholder" and not node.users:
                continue
            if node.op == "get_attr" and not self.consts:
                continue
            kind = {"placeholder": "in", "output": "out", "get_attr": "const"}.get(node.op, "op")
            label = node.name if node.op in ("placeholder", "output") else self._op_label(node, mods)
            if nconst.get(node.name):
                label += f" +{nconst[node.name]}c"
            st.nodes.append((_id(node.name), label, _shape_str(node), kind))
            for a in node.all_input_nodes:
                if a.op == "get_attr" and not self.consts:
                    continue
                if a.op == "placeholder" and not a.users:
                    continue
                st.edges.append((_id(a.name), _id(node.name), ""))
        self.stages.append(st)

    @staticmethod
    def _op_label(node, mods) -> str:
        from .dnn_torch_graph import _name
        if node.op == "call_module":
            m = mods.get(node.target)
            return type(m).__name__ if m is not None else str(node.target)
        if node.op == "call_method":
            return str(node.target)
        if node.op == "call_function":
            return _name(node.target)
        return node.op

    def ops(self, name: str, fg, note: str = "") -> None:
        """the graph API's ops: one per fx node, a wildcard where nothing mapped"""
        st = _Stage(name, note)
        for out, (kind, ins, _kw) in fg.ops.items():
            sub = self._tensor_str(fg, out)
            why = fg.wildcards.get(out)
            st.nodes.append((_id(out), kind if not why else f"wildcard ({why})", sub,
                             "wild" if why else "op"))
            for t in ins:
                if fg.graph.tensors.get(t, (None, None, False))[2] and not self.consts:
                    continue
                if t not in fg.ops:
                    st.nodes.append((_id(t), t, self._tensor_str(fg, t), "in"))
                st.edges.append((_id(t), _id(out), ""))
        seen, nodes = set(), []
        for n in st.nodes:                       # an input added once per reader
            if n[0] not in seen:
                seen.add(n[0])
                nodes.append(n)
        st.nodes = nodes
        self.stages.append(st)

    @staticmethod
    def _tensor_str(fg, t) -> str:
        d = fg.graph.tensors.get(t)
        return "" if d is None else f"{d[1]}[{','.join(str(x) for x in d[0])}]"

    def parts(self, name: str, fg, sup: list, cps: list, note: str = "") -> None:
        """the same ops, boxed by partition, each labelled with what the library made of it"""
        st = _Stage(name, note)
        impl = {id(p): c.impl for p, c in zip(sup, cps)}
        for p in fg.parts:
            ids = []
            for o in p.ops:
                kind = fg.ops.get(o, ("?", [], {}))[0]
                st.nodes.append((_id(o), kind, self._tensor_str(fg, o),
                                 "aie" if id(p) in impl else "no"))
                ids.append(_id(o))
            title = p.info + (f"  ->  {impl[id(p)]}" if id(p) in impl else
                              ("  (compile failed)" if p.info in fg.compile_failed else
                               "  (unsupported)" if not p.supported else "  (kept in torch)"))
            st.groups.append((title, ids, "aie" if id(p) in impl else "no"))
        inside = {o for p in fg.parts for o in p.ops}
        for out, (_k, ins, _kw) in fg.ops.items():
            for t in ins:
                if fg.graph.tensors.get(t, (None, None, False))[2] and not self.consts:
                    continue
                if t not in inside and t not in {n[0] for n in st.nodes}:
                    st.nodes.append((_id(t), t, self._tensor_str(fg, t), "in"))
                st.edges.append((_id(t), _id(out), ""))
        self.stages.append(st)

    def fused(self, name: str, fused, mods: dict, note: str = "") -> None:
        """the GraphModule torch executes: a node per compiled partition, everything else torch"""
        st = _Stage(name, note)
        sub = dict(fused.named_modules())
        for node in fused.graph.nodes:
            if node.op == "get_attr" and not self.consts:
                continue
            if node.op == "placeholder" and not node.users:
                continue
            pm = mods.get(str(node.target)) if node.op == "call_module" else None
            if pm is not None:
                label, kind = f"{node.target}: {pm.part.info}", "aie"
                detail = pm.cp.impl
            else:
                kind = {"placeholder": "in", "output": "out", "get_attr": "const"}.get(node.op, "torch")
                label = node.name if node.op in ("placeholder", "output") else self._op_label(node, sub)
                detail = _shape_str(node)
            st.nodes.append((_id(node.name), label, detail, kind))
            for a in node.all_input_nodes:
                if (a.op == "get_attr" and not self.consts) or (a.op == "placeholder" and not a.users):
                    continue
                st.edges.append((_id(a.name), _id(node.name), ""))
        self.stages.append(st)

    # ── rendering ────────────────────────────────────────────────────────
    def mermaid(self, st: _Stage) -> str:
        out = ["flowchart TD"]
        grouped = {i for _t, ids, _k in st.groups for i in ids}
        def emit(n, pad):
            nid, label, detail, kind = n
            text = _esc(label) + (f"<br/>{_esc(detail)}" if detail else "")
            out.append(f'{pad}{nid}["{text}"]:::{_STYLE[kind][0]}')
        for k, (title, ids, kind) in enumerate(st.groups):
            out.append(f'  subgraph g{k}["{_esc(title)}"]')
            for n in st.nodes:
                if n[0] in ids:
                    emit(n, "    ")
            out.append("  end")
        for n in st.nodes:
            if n[0] not in grouped:
                emit(n, "  ")
        for a, b, _l in dict.fromkeys(st.edges):
            out.append(f"  {a} --> {b}")
        out += [f"  classDef {c} {s}" for c, s in _CLASSDEF.items()]
        return "\n".join(out)

    def dot(self, st: _Stage) -> str:
        out = ["digraph g {", '  node [shape=box style="rounded,filled" fontname="Helvetica" fontsize=10];',
               '  edge [color="#888"];', "  rankdir=TB;"]
        grouped = {i for _t, ids, _k in st.groups for i in ids}
        def emit(n, pad):
            nid, label, detail, kind = n
            text = label.replace('"', "'") + (rf"\n{detail}" if detail else "")
            fill, fc = _STYLE[kind][1], _STYLE[kind][2]
            out.append(f'{pad}{nid} [label="{text}" fillcolor="{fill}" fontcolor="{fc}"];')
        for k, (title, ids, _kind) in enumerate(st.groups):
            out.append(f'  subgraph cluster{k} {{ label="{title.replace(chr(34), chr(39))}"; '
                       'style="rounded,dashed"; color="#888"; fontsize=10;')
            for n in st.nodes:
                if n[0] in ids:
                    emit(n, "    ")
            out.append("  }")
        for n in st.nodes:
            if n[0] not in grouped:
                emit(n, "  ")
        for a, b, _l in dict.fromkeys(st.edges):
            out.append(f"  {a} -> {b};")
        out.append("}")
        return "\n".join(out)

    def write(self, outdir) -> Path:
        """<outdir>/index.md (a Mermaid chart per stage) + stageN.dot, and .svg when dot is installed"""
        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        have_dot = shutil.which("dot")
        md = ["# `torch.compile(backend=\"aiednn\")`: the op graph at every stage", "",
              "| # | stage | nodes | |", "|---|---|---|---|"]
        for k, st in enumerate(self.stages, 1):
            md.append(f"| {k} | [{st.name}](#{k}-{st.name.replace(' ', '-')}) | {len(st.nodes)} | {st.note} |")
        for k, st in enumerate(self.stages, 1):
            stem = f"stage{k}_{st.name.replace(' ', '_').replace('+', '')}"
            (outdir / f"{stem}.dot").write_text(self.dot(st) + "\n")
            if have_dot:
                subprocess.run([have_dot, "-Tsvg", "-o", str(outdir / f"{stem}.svg"),
                                str(outdir / f"{stem}.dot")], check=False)
            md += ["", f"## {k} {st.name}", ""]
            if st.note:
                md.append(st.note + "")
            md += ["", "```mermaid", self.mermaid(st), "```", "",
                   f"[{stem}.svg]({stem}.svg)" if have_dot else f"[{stem}.dot]({stem}.dot)"]
        (outdir / "index.md").write_text("\n".join(md) + "\n")
        return outdir / "index.md"
