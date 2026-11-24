###############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
"""Group offload on the graph (design 2026-11-16-aiegraph-group-offload, §3.1 / §6, step S2).

A *group* is a set of whole TVM calls whose replacement steps only agree with TVM
**together** (attention: pack Q, pack K, pack V, run FA -> the merged O). Every call of
the group is either replaced by one step (it keeps its ``/* [k] */`` slot) or removed
(``/* [k] removed: group <name> */``). No call outside the group moves, so TVM's
storage plan still holds -- provided the checks below pass. They run on
``<stem>_graph.json`` (calls, tensor ids = ``tensors[i]`` of the driver, storage ids =
the ``buf_*`` the driver binds), never on generated C text:

placement   a step that reads goes to the earliest still-free group call that reads one
            of its inputs, after every input's producer and after the previous step;
            a write-only step goes to the first free group call after the previous step.
reads       a step may only read boundary inputs (produced outside the group, or graph
            constants); reading a tensor the group itself produces is an error.
liveness    an input must still be in its buffer at the step's slot: no kept call (or
            earlier step) between its producer and the slot writes that buffer
            ("overwritten buffer") -- the graph.json form of keep_layout_rt._untouched.
deletion    every tensor the group produces that a call outside the group (or the graph
            output) reads must be written by a step; otherwise the deleted intermediate
            has an external reader -- the graph.json form of keep_layout_rt._consumers.
early write a step writing T before T's original producer requires that no kept call in
            between touches T's buffer and no later step reads that buffer; writing it
            after an outside reader of T is an error as well.
in place    an "args" ABI step writing T away from T's producer must not read another tensor
            that lives in T's buffer (TVM reused the buffer once that tensor was dead -- the moved write
            would clobber its own input).
labels      an "args" ABI step may get other packed arguments (``ag.arg``): a tensor of the
            group seen through another shape (a layout that is only a relabel). A label of a
            group-internal tensor needs its producer to be a step and the buffer untouched by
            any kept call in between.

``partition.json`` records each group (``groups[]``: members with their action, steps
with slot, tensors and constants, ``io`` = boundary name / shape / dtype per input and
output, ``launch`` = what the main program launches for it); ``reset_groups`` takes a
whole group back out.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .errors import AiegraphError

__all__ = ["CallGraph", "GraphCall", "GroupTensor", "Group", "GroupStep", "GroupPlan",
           "group_step", "make_group", "plan_group", "attention_groups",
           "record_group", "recorded_groups", "reset_groups", "load_call_graph", "emit_steps"]

_STAGE = "group"


# ═══════════════════════════════════════════════════════════════════════════
#  graph.json at call level
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class GraphCall:
    call: int                   # execution-order index = "[NN]" in graph_driver.c
    node: int                   # graph.json node id
    symbol: str                 # tvmgen_default_fused_...
    ins: tuple                  # tensor ids read (canonical, __nop resolved)
    outs: tuple                 # tensor ids written

    @property
    def fn(self) -> str:
        return self.symbol.replace("tvmgen_default_fused_", "")


class CallGraph:
    """Calls, tensors and storage of one TVM graph executor JSON (TVM-free)."""

    def __init__(self, graph: dict, path=None):
        self.path = Path(path) if path else None
        self.graph = graph
        nodes, rp = graph["nodes"], graph["node_row_ptr"]
        self.sid = list(graph["attrs"]["storage_id"][1])
        self.shape = [tuple(s) for s in graph["attrs"]["shape"][1]]
        self.dtype = list(graph["attrs"]["dltype"][1])
        self.alias = {}                         # __nop output -> its input (same buffer)
        self.const = set()                      # tensor ids of null nodes (params, graph inputs)
        self.names = {}
        self.calls = []
        for i, n in enumerate(nodes):
            if n.get("op") != "tvm_op":
                self.const.add(rp[i])
                self.names[rp[i]] = n.get("name", "")
                continue
            ins = tuple(self.canon(rp[s] + k) for s, k, _ in n["inputs"])
            fn = n["attrs"]["func_name"]
            if fn == "__nop":
                self.alias[rp[i]] = ins[0]
                continue
            outs = tuple(rp[i] + k for k in range(int(n["attrs"].get("num_outputs", 1))))
            self.calls.append(GraphCall(len(self.calls), i, fn, ins, outs))
        self.heads = {self.canon(rp[s] + k) for s, k, _ in graph.get("heads", [])}
        self.producer = {t: c.call for c in self.calls for t in c.outs}
        self.readers = {}
        for c in self.calls:
            for t in c.ins:
                self.readers.setdefault(t, []).append(c.call)

    @classmethod
    def load(cls, path) -> "CallGraph":
        path = Path(path)
        if path.is_dir():
            g = next(path.glob("*_graph.json"), None)
            if g is None:
                raise AiegraphError(_STAGE, f"no *_graph.json in {path}",
                                    hint="convert the model first (ag.load(...).convert('a78'))")
            path = g
        if path.suffix == ".gz":
            import gzip
            return cls(json.loads(gzip.decompress(path.read_bytes())), path)
        return cls(json.loads(path.read_text()), path)

    def canon(self, t: int) -> int:
        while t in self.alias:
            t = self.alias[t]
        return t

    def storage(self, t: int) -> int:
        return self.sid[self.canon(t)]

    def touches(self, call: int) -> tuple:
        c = self.calls[call]
        return c.ins + c.outs

    def tensor(self, t: int, name: str = "") -> "GroupTensor":
        t = self.canon(t)
        shp = self.shape[t]
        return GroupTensor(t, shp, self.dtype[t], int(shp[-1]) if shp else 1,
                           name or f"tensors[{t}]", self.storage(t), self.producer.get(t))


def load_call_graph(workdir_or_json) -> CallGraph:
    return CallGraph.load(workdir_or_json)


# ═══════════════════════════════════════════════════════════════════════════
#  groups, steps, plans
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class GroupTensor:
    """A boundary tensor: ``tensors[tensor]`` of the driver, in storage ``sid``."""
    tensor: int
    shape: tuple
    dtype: str
    ld: int
    name: str
    sid: int
    producer: int = None        # producing call, None for a graph constant / input

    def __int__(self) -> int:
        return self.tensor

    def __str__(self) -> str:
        return f"tensors[{self.tensor}] {self.dtype}{list(self.shape)}"


@dataclass
class Group:
    name: str
    calls: tuple                # sorted call indices
    inputs: list                # GroupTensor read by the group, produced outside (activations)
    consts: list                # GroupTensor read by the group with no producer (params, inputs)
    outputs: list               # GroupTensor produced inside, read outside (or a graph output)
    internal: list              # GroupTensor produced inside, read only inside
    symbols: dict = field(default_factory=dict)     # call -> symbol
    # one-call weighted groups (callsite.py, S1): activation in / out, weight, geometry
    x: object = None
    y: object = None
    w: object = None
    info: dict = field(default_factory=dict)

    def __repr__(self) -> str:
        f = lambda ts: ", ".join(str(t) for t in ts) or "-"   # noqa: E731
        return (f"group {self.name}: calls {', '.join(f'[{c}]' for c in self.calls)}\n"
                f"  in:  {f(self.inputs)}\n  out: {f(self.outputs)}\n"
                f"  internal: {f(self.internal)}")


@dataclass(frozen=True)
class GroupStep:
    name: str
    reads: tuple = ()
    writes: tuple = ()
    consts: dict = field(default_factory=dict)


def group_step(name: str, reads=None, writes=None, consts=None) -> GroupStep:
    """A step as the group planner needs it (``ag.step`` objects work as well: anything
    with ``name``/``reads``/``writes`` whose items are tensor ids or carry ``.tensor``)."""
    if not isinstance(name, str) or not name.isidentifier():
        raise AiegraphError(_STAGE, f"step name {name!r}", hint="the C entry name of the step")
    lst = lambda x: () if x is None else tuple(x) if isinstance(x, (list, tuple)) else (x,)  # noqa
    return GroupStep(name, lst(reads), lst(writes), dict(consts or {}))


def _tid(x, who: str) -> int:
    t = getattr(x, "tensor", x)
    if isinstance(t, bool) or not isinstance(t, int):
        try:
            t = int(t)
        except (TypeError, ValueError):
            raise AiegraphError(_STAGE, f"{who}: {x!r} names no graph tensor",
                                hint="a group boundary tensor (g.inputs[i] / g.outputs[i]) "
                                     "or a tensor id") from None
    return t


@dataclass
class GroupPlan:
    group: Group
    steps: list                 # [{name, at, reads, writes, consts}]
    actions: dict               # call -> step name | "removed"
    launch: dict = None         # what the main program launches for the group (record only)

    @property
    def removed(self) -> list:
        return [c for c in self.group.calls if self.actions[c] == "removed"]

    @property
    def placement(self) -> dict:
        return {s["name"]: s["at"] for s in self.steps}

    def driver_comments(self) -> dict:
        """call -> the comment the generated driver carries at that call."""
        return {c: (f"/* [{c}] */" if a != "removed" else f"/* [{c}] removed: group {self.group.name} */")
                for c, a in self.actions.items()}

    def io(self) -> dict:
        """Boundary tensors by name / shape / dtype (every output: a group may have several)."""
        f = lambda ts: [{"name": t.name, "tensor": t.tensor, "shape": list(t.shape),   # noqa: E731
                         "dtype": t.dtype} for t in ts]
        return {"inputs": f(self.group.inputs), "consts": f(self.group.consts),
                "outputs": f(self.group.outputs)}

    def record(self) -> dict:
        g = self.group
        launch = dict(self.launch or {})
        launch.setdefault("steps", len(self.steps))
        return {"name": g.name, "calls": list(g.calls), "io": self.io(), "launch": launch,
                "members": [{"call": c, "symbol": g.symbols.get(c, ""),
                             "action": "removed" if a == "removed" else "step",
                             **({} if a == "removed" else {"step": a})}
                            for c, a in sorted(self.actions.items())],
                "steps": [dict(s) for s in self.steps],
                "inputs": [t.tensor for t in g.inputs], "outputs": [t.tensor for t in g.outputs]}

    def __repr__(self) -> str:
        lines = [f"group {self.group.name} plan:"]
        for c in self.group.calls:
            a = self.actions[c]
            lines.append(f"  {self.driver_comments()[c]}" + ("" if a == "removed" else f" {a}"))
        return "\n".join(lines)


#: tokens the driver generator fills in (not template parameters): the call index, the
#: step's bf16 weight in weights.bin as ``(const uint16_t *)(g_params + off)``
GEN_TOKENS = ("K", "WBF")
#: a template line that splits file-scope declarations (above) from the step's statements
STEP_MARK = "//@@ step"


def render_step(template: str, params: dict) -> tuple:
    """``(decl, [statement, ...])`` of an aiednn step template rendered with *params*
    (``@NAME@``; ``@K@`` / ``@WBF@`` are left for the driver generator). The text above a
    ``//@@ step`` line is file-scope C (emitted once per driver), the rest the step's
    statements, one per line."""
    import sys
    src = str(Path(__file__).resolve().parents[1])     # src/: aiednn sits beside aiegraph
    if src not in sys.path:
        sys.path.insert(0, src)
    import aiednn
    want = set(aiednn.params(template)) - set(GEN_TOKENS)
    kv = {k: v for k, v in dict(params or {}).items()}
    if set(kv) != want:
        raise AiegraphError(_STAGE, f"template {template}: missing params {sorted(want - set(kv))}, "
                            f"unknown {sorted(set(kv) - want)}")
    text = aiednn.render(template, **kv, **{k: f"@{k}@" for k in GEN_TOKENS
                                             if f"@{k}@" in aiednn.template(template)})
    lines = text.splitlines()
    marks = [i for i, ln in enumerate(lines) if ln.strip() == STEP_MARK]
    if len(marks) != 1:
        raise AiegraphError(_STAGE, f"template {template}: {len(marks)} '{STEP_MARK}' lines",
                            hint="exactly one: file-scope C above it, the step's statements below")
    raw = aiednn.template(template).splitlines()
    head = [ln for ln in raw[:[i for i, ln in enumerate(raw) if ln.strip() == STEP_MARK][0]] if "@" in ln
            and any(f"@{k}@" in ln for k in aiednn.params(template))]
    if head:
        raise AiegraphError(_STAGE, f"template {template}: parameter in the file-scope part: {head[0].strip()!r}",
                            hint="file-scope C is shared by every step of the driver: no @PARAM@ above the "
                                 f"'{STEP_MARK}' line")
    decl = "\n".join(lines[:marks[0]]).strip()
    body = [ln.strip() for ln in lines[marks[0] + 1:] if ln.strip()]
    return decl, body


def emit_steps(plan: GroupPlan, emit) -> GroupPlan:
    """Attach the driver code of every step (S3): ``emit`` maps a step name -- or is a callable
    taking the step record -- to ``{"c": [statement, ...], "tag": "..."}`` or to an aiednn
    template ``{"template": "aiednn/<category>/<file>", "params": {...}, "tag": "..."}``
    (``render_step``: file-scope declarations + statements). The statements are emitted
    verbatim at the step's call (``@K@`` -> the call index, ``@WBF@`` -> the step's bf16
    weight) between ``GRAPH_PROF_BEGIN/END`` (tag ``"cpu ..."`` for a step that runs on
    the A78, else ``"aie <tag>"``); ``arm_build.generate_driver`` refuses a group with a
    step that has none."""
    for s in plan.steps:
        e = emit(s) if callable(emit) else emit.get(s["name"])
        if isinstance(e, dict) and e.get("template"):
            decl, body = render_step(e["template"], e.get("params"))
            e = dict(e, c=body)
            s["template"] = e["template"]
            s["params"] = dict(e.get("params") or {})
            if decl:
                s["decl"] = decl
        if not isinstance(e, dict) or not e.get("c"):
            raise AiegraphError(_STAGE, f"group {plan.group.name}: no driver code for step {s['name']}",
                                hint='emit={step: {"template": "aiednn/<cat>/<file>", "params": {...}}} '
                                     'or {step: {"c": [...], "tag": ...}}')
        c = [e["c"]] if isinstance(e["c"], str) else list(e["c"])
        if not all(isinstance(x, str) and x.strip() for x in c):
            raise AiegraphError(_STAGE, f"group {plan.group.name}: step {s['name']}: c must be statements")
        if any("@WBF@" in x for x in c) and not s.get("weights"):
            raise AiegraphError(_STAGE, f"group {plan.group.name}: step {s['name']} uses @WBF@ "
                                f"but has no weights=", hint="ag.step(..., weights=g.w.as_bf16(...))")
        s["c"] = c
        s["tag"] = str(e.get("tag") or s["name"])
    return plan


def make_group(cg: CallGraph, calls, name: str = None) -> Group:
    """The group of these whole calls, with its boundary computed from the graph."""
    try:
        calls = tuple(sorted({int(c) for c in calls}))
    except (TypeError, ValueError):
        raise AiegraphError(_STAGE, f"group calls {calls!r}", hint="call indices [NN]") from None
    if not calls:
        raise AiegraphError(_STAGE, "empty group", hint="at least one TVM call")
    bad = [c for c in calls if not 0 <= c < len(cg.calls)]
    if bad:
        raise AiegraphError(_STAGE, f"group {name}: no call {bad} (graph has {len(cg.calls)})")
    name = name or f"g{calls[0]}"
    inside = set(calls)
    produced = [t for c in calls for t in cg.calls[c].outs]
    read = []
    for c in calls:
        for t in cg.calls[c].ins:
            if t not in read:
                read.append(t)
    ins = [cg.tensor(t) for t in read if cg.producer.get(t) not in inside and t not in cg.const]
    consts = [cg.tensor(t) for t in read if t in cg.const]
    outs, internal = [], []
    for t in produced:
        ext = t in cg.heads or any(r not in inside for r in cg.readers.get(t, []))
        (outs if ext else internal).append(cg.tensor(t))
    return Group(name, calls, ins, consts, outs, internal,
                 {c: cg.calls[c].symbol for c in calls})


def plan_group(cg: CallGraph, group: Group, steps) -> GroupPlan:
    """Place the steps on the group's calls and run every safety check (see module doc).
    Raises ``AiegraphError`` naming the call and tensor for the first violation."""
    steps = list(steps)
    if not steps:
        raise AiegraphError(_STAGE, f"group {group.name}: no steps")
    if len(steps) > len(group.calls):
        raise AiegraphError(_STAGE, f"group {group.name}: {len(steps)} steps > "
                            f"{len(group.calls)} calls", hint="one step per call at most")
    inside = set(group.calls)
    produced = {t.tensor for t in group.outputs + group.internal}
    pos = {c: i for i, c in enumerate(group.calls)}

    # ---- tensors each step names ---------------------------------------------------
    norm = []
    for s in steps:
        nm = getattr(s, "name", None)
        rd = [cg.canon(_tid(x, f"{nm} reads")) for x in getattr(s, "reads", ())]
        wr = [cg.canon(_tid(x, f"{nm} writes")) for x in getattr(s, "writes", ())]
        for t in rd:
            if t in produced:
                raise AiegraphError(_STAGE, f"group {group.name}: step {nm} reads tensors[{t}], "
                                    f"which call [{cg.producer[t]}] of the group produces "
                                    f"(it is not computed any more)",
                                    hint="steps read boundary inputs only; keep state in the kernel")
            if t not in cg.const and cg.producer.get(t) is None:
                raise AiegraphError(_STAGE, f"group {group.name}: step {nm} reads tensors[{t}], "
                                    f"which has no producer")
            if t not in cg.const and not any(t in cg.calls[c].ins for c in group.calls):
                raise AiegraphError(_STAGE, f"group {group.name}: step {nm} reads tensors[{t}], "
                                    f"which no call of the group reads",
                                    hint="g.inputs lists the boundary inputs")
        for t in wr:
            if t not in produced:
                raise AiegraphError(_STAGE, f"group {group.name}: step {nm} writes tensors[{t}], "
                                    f"which the group does not produce",
                                    hint="steps write group outputs only (g.outputs)")
        norm.append((nm, rd, wr, dict(getattr(s, "consts", {}) or {}), s))
    names = [n for n, *_ in norm]
    if len(set(names)) != len(names):
        raise AiegraphError(_STAGE, f"group {group.name}: step names repeat {names}")

    # ---- placement -----------------------------------------------------------------
    taken, placed, prev = set(), [], -1
    for nm, rd, wr, consts, st in norm:
        acts = [t for t in rd if t not in cg.const]
        after = max([prev] + [cg.producer[t] for t in acts])
        pin = getattr(st, "at", None)
        if pin is not None:
            if pin not in inside or pin in taken or pin <= after:
                raise AiegraphError(_STAGE, f"group {group.name}: step {nm} pinned at [{pin}]: "
                                    + ("not a call of the group" if pin not in inside else
                                       "already taken" if pin in taken else
                                       f"not after call [{after}] (previous step / input producer)"),
                                    hint="at= is a free group call in data order")
            cand = [pin]
        else:
            cand = [c for c in group.calls if c > after and c not in taken]
            if acts:
                cand = [c for c in cand if any(t in cg.calls[c].ins for t in acts)]
        if not cand:
            raise AiegraphError(_STAGE, f"group {group.name}: no free call for step {nm} "
                                f"after call [{after}]",
                                hint="steps go in data order; a reading step needs a group call "
                                     "that reads its input")
        at = cand[0]
        taken.add(at)
        rec = {"name": nm, "at": at, "reads": rd, "writes": wr, "consts": consts}
        if getattr(st, "args", None) is not None:
            rec["args"] = [a.record() for a in st.args]
        placed.append(rec)
        prev = at

    kept = [c for c in range(len(cg.calls)) if c not in inside]

    # ---- liveness of every read (overwritten buffer) --------------------------------
    for s in placed:
        for t in s["reads"]:
            if t in cg.const:
                continue
            sid, p = cg.storage(t), cg.producer[t]
            for c in kept:
                if p < c < s["at"]:
                    hit = [o for o in cg.calls[c].outs if o != t and cg.storage(o) == sid]
                    if hit:
                        raise AiegraphError(
                            _STAGE, f"group {group.name}: overwritten buffer -- step {s['name']} "
                            f"at [{s['at']}] reads tensors[{t}] (storage {sid}), but call [{c}] "
                            f"{cg.calls[c].fn} writes tensors[{hit[0]}] into the same buffer first",
                            hint="the input is dead by then in TVM's storage plan; read it at an "
                                 "earlier call")
            for w in placed:
                if w["at"] < s["at"]:
                    hit = [o for o in w["writes"] if cg.storage(o) == sid and o != t]
                    if hit:
                        raise AiegraphError(
                            _STAGE, f"group {group.name}: early-write conflict -- step {w['name']} "
                            f"at [{w['at']}] writes tensors[{hit[0]}] into storage {sid} before step "
                            f"{s['name']} at [{s['at']}] reads tensors[{t}] from it")

    # ---- deletion: no external reader of a tensor no step writes ----------------------
    written = {t for s in placed for t in s["writes"]}
    for t in group.outputs:
        if t.tensor not in written:
            ext = [r for r in cg.readers.get(t.tensor, []) if r not in inside]
            who = (f"call [{ext[0]}] {cg.calls[ext[0]].fn}" if ext else "the graph output")
            raise AiegraphError(
                _STAGE, f"group {group.name}: external reader -- tensors[{t.tensor}] from call "
                f"[{t.producer}] {cg.calls[t.producer].fn} is read by {who} outside the group, "
                f"but no step writes it",
                hint="add the reader to the group, or a step that writes this tensor")

    # ---- early / late write --------------------------------------------------------------
    for s in placed:
        for t in s["writes"]:
            sid, p = cg.storage(t), cg.producer[t]
            lo, hi = sorted((s["at"], p))
            for c in kept:
                if lo < c < hi and any(cg.storage(x) == sid for x in cg.touches(c)):
                    raise AiegraphError(
                        _STAGE, f"group {group.name}: early-write conflict -- step {s['name']} "
                        f"writes tensors[{t}] at [{s['at']}] instead of [{p}], but call [{c}] "
                        f"{cg.calls[c].fn} touches its buffer (storage {sid}) in between",
                        hint="write the output at (or right before) its original producer")
            late = [r for r in cg.readers.get(t, []) if r not in inside and r < s["at"]]
            if late:
                raise AiegraphError(_STAGE, f"group {group.name}: step {s['name']} writes "
                                    f"tensors[{t}] at [{s['at']}], after its reader [{late[0]}]")
        # a later step must not read a buffer this step has already overwritten
        # (checked above from the reader's side)

    # ---- in place: a moved write must not clobber what the step itself reads -------------
    by_step = {nm: st for nm, *_, st in norm}
    for s in placed:
        st = by_step[s["name"]]
        if not getattr(st, "call", None):
            continue                        # an emit= step reads its inputs before it writes
        # "args" ABI: a kernel that streams its output while still reading its operands
        own_in, own_out = set(cg.calls[s["at"]].ins), set(cg.calls[s["at"]].outs)
        if s.get("args"):
            packed = [cg.canon(a["tensor"]) for a in s["args"]]
            written = [t for t in packed if t in s["writes"] or t in own_out]
            early = {cg.canon(a["tensor"]) for a in s["args"] if a.get("early")}
            reads = (set(packed) - set(written) - early) | (set(s["reads"]) - early)
        else:
            written, reads = list(s["writes"]) + sorted(own_out), own_in | set(s["reads"])
        for t in written:
            # TVM never puts a call's input in its own output buffer: only moved writes and
            # arguments the call did not have can clash
            cand = reads if cg.producer.get(t) != s["at"] else reads - own_in
            sid = cg.storage(t)
            clash = sorted(r for r in cand - {t} if r not in cg.const and cg.storage(r) == sid)
            if clash:
                raise AiegraphError(
                    _STAGE, f"group {group.name}: in-place hazard -- step {s['name']} at "
                    f"[{s['at']}] writes tensors[{t}] (storage {sid}, produced by call "
                    f"[{cg.producer.get(t)}]) and reads tensors[{clash[0]}] from the same buffer",
                    hint="TVM reused the buffer of a dead input; keep the call that writes it")

    actions = {c: "removed" for c in group.calls}
    for s in placed:
        actions[s["at"]] = s["name"]

    # ---- labels: packed arguments the "args" ABI step gets instead of the call's own -------
    for s in placed:
        st = by_step[s["name"]]
        if not s.get("args"):
            continue
        own = set(cg.calls[s["at"]].ins) | set(cg.calls[s["at"]].outs)
        for a in s["args"]:
            t = cg.canon(a["tensor"])
            if not 0 <= t < len(cg.shape):
                raise AiegraphError(_STAGE, f"group {group.name}: step {s['name']}: no tensors[{t}]")
            n = 1
            for d in cg.shape[t]:
                n *= int(d)
            if "shape" in a:
                m = 1
                for d in a["shape"]:
                    m *= int(d)
                if m != n:
                    raise AiegraphError(_STAGE, f"group {group.name}: step {s['name']}: label "
                                        f"{a['shape']} of tensors[{t}] {list(cg.shape[t])} changes the "
                                        f"element count ({m} != {n})")
            if t in own or t in cg.const:
                continue
            if t in s["writes"]:
                continue                    # an output written here (checked as an early write)
            p = cg.producer.get(t)
            if p is None:
                raise AiegraphError(_STAGE, f"group {group.name}: step {s['name']}: tensors[{t}] "
                                    f"has no producer")
            if p not in inside:
                if not any(t in cg.calls[c].ins for c in group.calls):
                    raise AiegraphError(_STAGE, f"group {group.name}: step {s['name']} gets "
                                        f"tensors[{t}], which no call of the group reads")
                ok_from = p
            else:
                if actions[p] == "removed":
                    raise AiegraphError(_STAGE, f"group {group.name}: step {s['name']} gets "
                                        f"tensors[{t}] of removed call [{p}] (never computed)")
                ok_from = p
            if not ok_from < s["at"]:
                raise AiegraphError(_STAGE, f"group {group.name}: step {s['name']} at [{s['at']}] "
                                    f"gets tensors[{t}] produced later, at [{p}]")
            sid = cg.storage(t)
            for c in range(ok_from + 1, s["at"]):
                if c in inside:
                    continue
                hit = [o for o in cg.calls[c].outs if o != t and cg.storage(o) == sid]
                if hit:
                    raise AiegraphError(_STAGE, f"group {group.name}: overwritten buffer -- step "
                                        f"{s['name']} at [{s['at']}] gets tensors[{t}] (storage {sid}), "
                                        f"but call [{c}] {cg.calls[c].fn} writes tensors[{hit[0]}] "
                                        f"into the same buffer first")
    return GroupPlan(group, placed, actions)


# ═══════════════════════════════════════════════════════════════════════════
#  attention cores by data flow on the call graph (S3 maps them to ONNX)
# ═══════════════════════════════════════════════════════════════════════════

def attention_groups(cg: CallGraph) -> list:
    """Every bmm -> softmax -> bmm core with its Q/K/V layout calls and the O merge:
    ``[q_t, k_t, qk, softmax, v_t, pv, merge]`` as one ``Group`` (graph order).
    Q/K/V/O are the group's boundary; scores, P and the head layouts are internal."""
    out = []
    for c in cg.calls:
        if "nn_softmax" not in c.fn:
            continue
        acts = lambda call: [t for t in call.ins if t not in cg.const]   # noqa: E731
        qk = cg.producer.get(acts(c)[0]) if acts(c) else None
        if qk is None or "batch_matmul" not in cg.calls[qk].fn:
            continue
        pv = [r for r in cg.readers.get(c.outs[0], []) if "batch_matmul" in cg.calls[r].fn]
        if len(pv) != 1:
            continue
        pv = pv[0]
        mem = {qk, c.call, pv}
        mem |= {cg.producer[t] for t in acts(cg.calls[qk]) if t in cg.producer}
        mem |= {cg.producer[t] for t in acts(cg.calls[pv]) if t in cg.producer and cg.producer[t] != c.call}
        merge = cg.readers.get(cg.calls[pv].outs[0], [])
        if len(merge) == 1 and "reshape" in cg.calls[merge[0]].fn:
            mem.add(merge[0])
        out.append(make_group(cg, mem, f"attn{len(out)}"))
    return out


# ═══════════════════════════════════════════════════════════════════════════
#  partition.json
# ═══════════════════════════════════════════════════════════════════════════

def _part(workdir) -> Path:
    return Path(workdir) / "layers" / "partition.json"


def _read(workdir) -> dict:
    p = _part(workdir)
    return json.loads(p.read_text()) if p.is_file() else {"layers": []}


def recorded_groups(workdir) -> list:
    return list(_read(workdir).get("groups", []))


# record_group's last write per file: the exact text, the doc it encodes (never handed out) and the
# indent=2 text of each top-level value / each group.  json.dumps(indent=2) runs the pure-Python
# encoder (~17 ms per 0.7 MB doc), and a UNet build records 139 groups one by one (8 s); with the
# cache only the new group is encoded.  Reused only when the file still holds exactly that text.
_REC_CACHE: dict = {}


def _ind(text: str, n: int) -> str:
    return text.replace("\n", "\n" + " " * n)


def _dump_cached(doc: dict, parts: dict, gtexts: dict) -> str:
    """== ``json.dumps(doc, indent=2)``, reusing *parts* (top-level key -> value text) and
    *gtexts* (group name -> group text); fills both."""
    if not doc:
        return "{}"
    items = []
    for k, v in doc.items():
        if k == "groups":
            names = [g.get("name") if isinstance(g, dict) else None for g in v] if isinstance(v, list) else []
            if not v or None in names or len(set(names)) != len(names):
                vt = _ind(json.dumps(v, indent=2), 2)      # empty / odd / duplicate names: in full
                items.append("  " + json.dumps(k) + ": " + vt)
                continue
            for g in v:
                if g["name"] not in gtexts:
                    gtexts[g["name"]] = json.dumps(g, indent=2)
            vt = "[\n" + ",\n".join("    " + _ind(gtexts[g["name"]], 4) for g in v) + "\n  ]"
        else:
            if k not in parts:
                parts[k] = _ind(json.dumps(v, indent=2), 2)
            vt = parts[k]
        items.append("  " + json.dumps(k) + ": " + vt)
    return "{\n" + ",\n".join(items) + "\n}"


def record_group(workdir, plan: GroupPlan, source: str = None) -> dict:
    """Write (or replace) the group in ``layers/partition.json``; a call may belong to
    one group only."""
    p = _part(workdir)
    text = p.read_text() if p.is_file() else None
    hit = _REC_CACHE.get(str(p.resolve()))
    if hit is not None and text is not None and hit[0] == text:
        _, doc, parts, gtexts = hit
    else:
        doc = json.loads(text) if text is not None else {"layers": []}
        parts, gtexts = {}, {}
    groups = [g for g in doc.get("groups", []) if g["name"] != plan.group.name]
    mine = set(plan.group.calls)
    for g in groups:
        both = sorted(mine & set(g["calls"]))
        if both:
            raise AiegraphError(_STAGE, f"group {plan.group.name}: calls {both} already belong "
                                f"to group {g['name']}",
                                hint=f"offload_reset('group:{g['name']}') first")
    rec = plan.record()
    if source is not None:
        rec["source"] = str(source)
    rtext = json.dumps(rec, indent=2)
    groups.append(json.loads(rtext))            # the cached doc never aliases the caller's rec
    doc["groups"] = sorted(groups, key=lambda g: g["calls"][0])
    gtexts = {n: t for n, t in gtexts.items() if n != plan.group.name}
    gtexts[plan.group.name] = rtext
    out = _dump_cached(doc, parts, gtexts) + "\n"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(out)
    _REC_CACHE.clear()                          # one file per process is the normal case
    _REC_CACHE[str(p.resolve())] = (out, doc, parts, gtexts)
    return rec


def reset_groups(workdir, names=None, calls=None) -> list:
    """Take whole groups out of partition.json: all (``names=None, calls=None``), the
    named ones, and any group with a member in *calls* (a group is never half-reset).
    Returns the names removed."""
    doc = _read(workdir)
    groups = doc.get("groups", [])
    if not groups:
        return []
    names = None if names is None else {str(n).removeprefix("group:") for n in names}
    calls = None if calls is None else set(calls)
    if names is not None:
        unknown = names - {g["name"] for g in groups}
        if unknown:
            raise AiegraphError(_STAGE, f"no recorded group {sorted(unknown)}",
                                hint=f"recorded: {[g['name'] for g in groups]}")
    every = names is None and calls is None
    drop = [g for g in groups if every or (names and g["name"] in names)
            or (calls and set(g["calls"]) & calls)]
    if not drop:
        return []
    doc["groups"] = [g for g in groups if g not in drop]
    _part(workdir).write_text(json.dumps(doc, indent=2) + "\n")
    return [g["name"] for g in drop]
