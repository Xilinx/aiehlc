"""``torch.compile(model, backend="aiednn")``: run PyTorch graphs through aiednn.

Dynamo hands the backend an fx graph. The backend splits it into regions aiednn can run and regions it
cannot (``CapabilityBasedPartitioner``), compiles each supported region to aiednn primitives and calls
them from the generated module; everything else stays torch code. Support is the lowering's own
verdict -- a region whose lowering fails hands the failing node back to torch and the graph is
re-partitioned -- the rule of the graph API ("create failing means unsupported",
doc/design/aiednn_framework_api.md section 2).

Interim inner part: a region is lowered with ``dnn_torch.lower`` (fx -> aiednn primitives; the MLIR
pipeline with ``mlir``) and run with ``run_ctypes`` on the workstation build of the library
(``libaiednn.so``: the AIE kernels' C references stand in for the array -- workstation emulation).
A region takes one tensor input (dnn_torch's graphs have one); other regions stay torch code. When the
partition graph API (aiednn_graph.h) lands, ``_compile_region`` switches to graph_add_op /
get_partitions / partition_compile behind the same interface.

Flags: ``aiegraph.backends.aiednn`` (benchmark, mode, mlir, allow_ref).
"""
from __future__ import annotations

import copy
import re

import numpy as np
import torch

from . import _env
from .errors import AiegraphError

_LIB = None
_NODE = re.compile(r"\[dnn_torch\] (\w+):|dnn_torch: (\w+):")


def _lib():
    global _LIB
    if _LIB is None:
        from .dnn_ctypes import AiednnLib
        _LIB = AiednnLib.build(_env.repo_root() / "worklocal" / "aiednn" / "torch_backend_lib")
    return _LIB


def _flags():
    from .backends import aiednn as flags
    return flags


class _Region:
    """one supported region, lowered once. Called with the fused_* call's arguments: *act* is the
    position of its one activation, the others are constants (get_attr in the caller) bound into the
    lowering. Returns what the submodule returns: one tensor, or a tuple when Dynamo keeps an
    intermediate alive (served from the kept buffers by node name)."""

    def __init__(self, sub, args, act):
        from torch.fx.experimental.optimization import fuse

        from . import dnn_torch as dt
        fl = _flags()
        self.sub, self.act = sub, act
        gm = copy.deepcopy(sub).eval()
        for i, ph in enumerate([n for n in gm.graph.nodes if n.op == "placeholder"]):
            if i != act:                       # a constant input -> an attribute of the region
                gm.register_buffer(f"_aiednn_in{i}", args[i].detach())
                with gm.graph.inserting_after(ph):
                    ph.replace_all_uses_with(gm.graph.get_attr(f"_aiednn_in{i}"))
                gm.graph.erase_node(ph)
        gm.recompile()
        out = next(n for n in gm.graph.nodes if n.op == "output").args[0]
        self.outs = [n.name for n in out] if isinstance(out, (tuple, list)) else None
        if self.outs and fl.mlir:
            raise AiegraphError("dnn_torch_backend", "a region with several outputs under mlir")
        # Conv+BN folded without a re-trace (no_trace): node names stay the partition's, so a
        # lowering error names a node the partitioner knows
        gm = fuse(gm, no_trace=True)
        self.g = dt.lower(gm, args[act], fl.mode, "torch_region", gm=gm, fuse=not fl.mlir)
        if fl.mlir:
            from .dnn_mlir import optimize
            self.g = optimize(self.g)
        self.impls = None        # implementation per primitive, after the first call
        self.on_torch = False    # allow_ref = False and every primitive ran on ref:

    def _tensor(self, buf, tid, like):
        shape = tuple(self.g.tensors[tid].shape)
        return torch.from_numpy(np.ascontiguousarray(buf, np.float32).reshape(shape)).to(like.device, like.dtype)

    def __call__(self, *args):
        if self.on_torch:
            return self.sub(*args)
        from . import dnn_torch as dt
        lib, x = _lib(), args[self.act]
        lib.so.aiednn_set_autotune(1 if _flags().benchmark else 0)
        y, impls, bufs = dt.run_ctypes(self.g, lib, x.detach().cpu().numpy().astype(np.float32), keep=True)
        if self.impls is None and not _flags().allow_ref and all(i.startswith("ref:") for i in impls):
            self.on_torch = True
            return self.sub(*args)
        self.impls = impls
        if self.outs is None:
            return self._tensor(y, self.g.output, x)
        return tuple(self._tensor(bufs[self.g.nodes[n]], self.g.nodes[n], x) for n in self.outs)


def _freeze_params(gm, example_inputs):
    """parameters Dynamo lifted to graph inputs -> get_attr constants (the region then has one input)"""
    import torch
    for i, node in enumerate([n for n in gm.graph.nodes if n.op == "placeholder"]):
        t = example_inputs[i] if i < len(example_inputs) else None
        if isinstance(t, torch.nn.Parameter) or "_parameters_" in node.name or "_buffers_" in node.name:
            name = f"_aiednn_const{i}"
            gm.register_buffer(name, t.detach() if isinstance(t, torch.Tensor) else t)
            with gm.graph.inserting_after(node):
                ga = gm.graph.get_attr(name)
            node.replace_all_uses_with(ga)   # the placeholder stays: the call signature is Dynamo's
    gm.recompile()


def _region_inputs(fused, example_inputs) -> dict:
    """every fused_* call's real arguments (calibration and shapes use them) and which are constants"""
    import torch.fx as fx
    seen = {}

    class _Rec(fx.Interpreter):
        def run_node(self, n):
            if n.op == "call_module" and str(n.target).startswith("fused_"):
                const = [isinstance(a, fx.Node) and a.op == "get_attr" for a in n.args]
                seen[str(n.target)] = (tuple(self.fetch_args_kwargs_from_env(n)[0]), const)
            return super().run_node(n)

    _Rec(fused).run(*example_inputs)
    return seen


def _partition(gm, unsupported):
    import torch.fx as fx
    from torch.fx.passes.infra.partitioner import CapabilityBasedPartitioner
    from torch.fx.passes.operator_support import OperatorSupport

    class _Support(OperatorSupport):
        def is_node_supported(self, submodules, node):
            return node.op in ("call_module", "call_function", "call_method") and node.name not in unsupported

    work = fx.GraphModule(gm, copy.deepcopy(gm.graph))
    part = CapabilityBasedPartitioner(work, _Support(), allows_single_node_partition=True)
    return part.fuse_partitions(part.propose_partitions())


def _graph_api() -> bool:
    """the partition graph API is in this tree (include/aiednn_graph.h + aiegraph.dnn_graph)"""
    try:
        from . import dnn_graph  # noqa: F401
    except ImportError:
        return False
    return (_env.repo_root() / "include" / "aiednn_graph.h").is_file()


_GLIB = None


def _graph_lib():
    global _GLIB
    if _GLIB is None:
        from .dnn_graph import build_graph_lib
        _GLIB = build_graph_lib(_env.repo_root() / "worklocal" / "aiednn" / "torch_backend_graphlib")
    return _GLIB


class _PartModule(torch.nn.Module):
    """one supported graph-API partition in place of its fx nodes: torch tensors in, the compiled
    partition executed on their buffers, torch tensors out (the submodule's output structure)"""

    def __init__(self, fg, part, cp, args, outs):
        super().__init__()
        self.fg, self.part, self.cp = fg, part, cp
        if _flags().benchmark:          # time every accepting implementation once, keep the fastest
            ranked = [r for r in self.cp.find(1) if r[4] == 0]
            if len(ranked) > 1:        # one candidate (a copy, a lone ref:) is already what runs
                self.cp.set_impl(ranked[0][0])
        self.args, self.outs, self.single = args, outs, len(outs) == 1

    def forward(self, *args):
        from .dnn_ctypes import DT
        g, env = self.fg.graph, dict(zip(self.args, args))
        ins = {}
        for t in self.part.inputs:
            ins[t] = self.fg.consts[t] if t in self.fg.consts else np.ascontiguousarray(
                env[t].detach().cpu().numpy())
        outs = {t: np.empty(g.tensors[t][0], DT[g.tensors[t][1]][1]) for t in self.part.outputs}
        self.cp.execute(ins, outs)
        like = args[0] if args else None
        def out(t):   # a node whose integer getitem became slice + reshape: the slice's buffer, reshaped
            return outs[t] if t in outs else outs[t + ".slice"].reshape(g.tensors[t][0])
        res = [torch.from_numpy(out(t)).to(like.device if like is not None else "cpu") for t in self.outs]
        return res[0] if self.single else tuple(res)


def _boundary(nodes) -> tuple:
    """(input names, output names) of a node group, in the order fuse_as_graphmodule gives its
    submodule's placeholders and outputs: inputs by first use over the topologically sorted nodes
    (args, then kwargs), outputs = nodes with a user outside the group"""
    import torch.fx as fx
    from torch.fx.passes.utils.fuser_utils import topo_sort
    nodes = topo_sort(nodes)
    inside, ins = set(nodes), []
    for n in nodes:
        def see(a):
            if a not in inside and a.name not in ins:
                ins.append(a.name)
            return a
        fx.node.map_arg((n.args, n.kwargs), see)
    outs = [n.name for n in nodes if any(u not in inside for u in n.users)]
    return ins, outs


def _perm_of(node):
    """the permutation of a permute node (method / torch / aten), else None"""
    if node.op not in ("call_method", "call_function"):
        return None
    from .dnn_torch_graph import _name
    if (node.target if node.op == "call_method" else _name(node.target)) != "permute":
        return None
    perm = node.args[1:] if len(node.args) > 1 and not isinstance(node.args[1], (list, tuple)) else node.args[1]
    return [int(p) for p in perm]


def _sink_permute(gm) -> int:
    """mul / add of a permute's result by a constant -> the op above the permute, the constant permuted
    back (--aiednn-sink-permute on the fx side): convnext's layer scale then sits right after its GEMM,
    where the graph API fuses it as a post-op. Returns how many ops moved."""
    from .dnn_torch_graph import _name
    moved = 0
    for node in list(gm.graph.nodes):
        if node.op not in ("call_function", "call_method"):
            continue
        name = node.target if node.op == "call_method" else _name(node.target)
        if name not in ("mul", "add") or len(node.args) != 2 or node.kwargs:
            continue
        a, b = node.args
        p, c = (a, b) if _perm_of(a) is not None else (b, a)
        perm = _perm_of(p) if hasattr(p, "op") else None
        if perm is None or len(p.users) != 1 or not hasattr(c, "op") or c.op != "get_attr":
            continue
        t = gm
        for part in c.target.split("."):
            t = getattr(t, part)
        r = len(perm)
        if not isinstance(t, torch.Tensor) or t.dim() > r:
            continue
        inv = [0] * r
        for k, q in enumerate(perm):
            inv[q] = k
        tc = t.reshape((1,) * (r - t.dim()) + tuple(t.shape)).permute(inv).contiguous()
        cname = f"_aiednn_sunk{moved}"
        gm.register_buffer(cname, tc)
        y = p.args[0]
        with gm.graph.inserting_before(p):
            cn = gm.graph.get_attr(cname)
            op = gm.graph.call_function(node.target, (y, cn)) if node.op == "call_function" else \
                gm.graph.call_method(node.target, (y, cn))
        p.replace_input_with(y, op)
        node.replace_all_uses_with(p)
        gm.graph.erase_node(node)
        moved += 1
    if moved:
        gm.graph.eliminate_dead_code()
        gm.recompile()
    return moved


def _inline_mha(gm, example_inputs) -> int:
    """nn.MultiheadAttention submodules (Dynamo keeps them as one call_module: vit) -> their decomposition
    (dnn_torch.mha_module: Linear, split, sdpa, Linear) traced and inlined in place, so the graph API sees
    the GEMMs and the attention. Self-attention only (q, k, v the same node), no masks; returns calls moved."""
    import operator

    import torch
    import torch.fx as fx

    from .dnn_torch import mha_module
    moved = 0
    for node in list(gm.graph.nodes):
        if node.op != "call_module":
            continue
        m = gm.get_submodule(node.target)
        if not isinstance(m, torch.nn.MultiheadAttention) or not (m.batch_first and m._qkv_same_embed_dim):
            continue
        q = node.args[0] if node.args else node.kwargs.get("query")
        kv = [node.args[i] if len(node.args) > i else node.kwargs.get(("query", "key", "value")[i]) for i in (1, 2)]
        if any(v is not q for v in kv) or node.kwargs.get("attn_mask") is not None or \
                node.kwargs.get("key_padding_mask") is not None:
            continue
        dec = mha_module(m)
        name = node.target.replace(".", "_") + "_dec"
        gm.add_submodule(name, dec)
        sub = fx.symbolic_trace(dec, concrete_args={"need_weights": False, "attn_mask": None, "key_padding_mask": None})
        ph = [n for n in sub.graph.nodes if n.op == "placeholder"]
        env = {p: (q if i < 3 else (False if i == 3 else None)) for i, p in enumerate(ph)}   # need_weights, masks
        with gm.graph.inserting_before(node):
            for n in sub.graph.nodes:
                if n.op == "placeholder":
                    continue
                if n.op == "output":
                    out = n.args[0]
                    break
                new = gm.graph.node_copy(n, lambda a: env[a])
                if new.op in ("call_module", "get_attr"):
                    new.target = f"{name}.{n.target}"
                env[n] = new
        res = env[out[0]] if isinstance(out, (tuple, list)) else env[out]
        for u in list(node.users):   # (out, weights)[0] -> the decomposition's output; [1] is None
            if u.op == "call_function" and u.target is operator.getitem and u.args[1] == 0:
                u.replace_all_uses_with(res)
                gm.graph.erase_node(u)
        if not node.users:
            gm.graph.erase_node(node)
            moved += 1
    if moved:
        gm.graph.lint()
        gm.recompile()
        from torch.fx.passes.shape_prop import ShapeProp
        ShapeProp(gm).propagate(*example_inputs)   # meta for the inlined nodes (_split_to_narrow reads it)
    return moved


def _split_to_narrow(gm) -> int:
    """x.split / x.chunk whose pieces are only picked by integer getitem -> one torch.narrow per piece.
    The split's tuple is no tensor, so a partition reading a piece would have the tuple, not x, at its
    boundary; narrow reads x itself. Sizes from Dynamo's example values (node.meta); returns pieces moved."""
    import operator

    import torch
    moved = 0
    for node in list(gm.graph.nodes):
        name = node.target if isinstance(node.target, str) else getattr(node.target, "__name__", "")
        name = name.split(".")[0]   # aten.split.Tensor / split_with_sizes.default (PT2E graphs)
        if node.op not in ("call_method", "call_function") or name not in ("split", "chunk", "split_with_sizes"):
            continue
        ev = node.meta.get("example_value") or node.meta.get("val") or node.meta.get("tensor_meta")
        if not isinstance(ev, (tuple, list)) or not node.users or \
                not all(u.target is operator.getitem and isinstance(u.args[1], int) for u in node.users):
            continue
        x = node.args[0]
        dim = node.kwargs.get("dim", node.args[2] if len(node.args) > 2 else 0)
        dim %= len(ev[0].shape)
        offs = [0]
        for t in ev:
            offs.append(offs[-1] + int(t.shape[dim]))
        for u in list(node.users):
            i = u.args[1]
            with gm.graph.inserting_before(u):
                n = gm.graph.call_function(torch.narrow, (x, dim, offs[i], int(ev[i].shape[dim])))
            n.meta.update(u.meta)
            u.replace_all_uses_with(n)
            gm.graph.erase_node(u)
            moved += 1
        gm.graph.erase_node(node)
    if moved:
        gm.recompile()
    return moved


def _select_to_narrow(gm) -> int:
    """x.select(dim, i) (aten.select.int: PT2E graphs) -> narrow(x, dim, i, 1) + squeeze(dim), two real fx nodes.
    FxGraph would otherwise add a helper slice op plus a reshape for the one node, and when the two land in
    different partitions both would claim that node. Returns nodes rewritten."""
    import torch
    n_done = 0
    for node in list(gm.graph.nodes):
        if node.op != "call_function" or getattr(node.target, "__name__", "").split(".")[0] != "select" or \
                len(node.args) < 3 or not isinstance(node.args[1], int) or not isinstance(node.args[2], int):
            continue
        x, dim, idx = node.args[:3]
        with gm.graph.inserting_before(node):
            nar = gm.graph.call_function(torch.narrow, (x, dim, idx, 1))
            sq = gm.graph.call_function(torch.squeeze, (nar, dim))
        sq.meta.update(node.meta)
        node.replace_all_uses_with(sq)
        gm.graph.erase_node(node)
        n_done += 1
    if n_done:
        gm.recompile()   # FxGraph's own shape propagation gives the new nodes their meta
    return n_done


def _group_nodes(fg, part, by_name) -> list:
    """the fx nodes a partition replaces: its ops' nodes (a helper op such as 'x.slice' belongs to node
    x), plus the nodes FxGraph folded into it -- a GQA repeat_interleave read only inside, a split whose
    every piece is -- so the fused submodule's inputs are the tensors the partition reads"""
    names = list(dict.fromkeys(o if o in by_name else o.rsplit(".", 1)[0] for o in part.ops))
    nodes = [by_name[n] for n in names if n in by_name]
    inside = set(nodes)
    grew = True
    while grew:
        grew = False
        for n in list(inside):
            for a in n.all_input_nodes:
                if a not in inside and (a.name in fg.alias or a.name not in fg.graph.tensors) and \
                        a.op not in ("placeholder", "get_attr") and a.users and set(a.users) <= inside:
                    inside.add(a)
                    nodes.append(a)
                    grew = True
    return nodes


def _graph_backend(gm, example_inputs, tr=None):
    """partitioning by the library: every fx node becomes a graph op (dnn_torch_graph), the graph API
    returns partitions, each supported one replaces its nodes with one compiled aiednn primitive"""
    from torch.fx.passes.utils.fuser_utils import fuse_by_partitions

    from .dnn_torch_graph import FxGraph
    from torch.fx.experimental.optimization import fuse
    # Conv+BN folded on the fx side as dnn_torch.prepare does (the graph API does not fold BN into a
    # conv), without a re-trace so the call signature stays Dynamo's
    gm = fuse(gm, no_trace=True)
    if tr:
        tr.fx("conv+bn", gm, "BN folded into its conv (torch.fx.experimental.optimization.fuse)")
    moved = (_sink_permute(gm), _inline_mha(gm, example_inputs), _split_to_narrow(gm), _select_to_narrow(gm))
    fg = FxGraph(_graph_lib(), gm, example_inputs, fuse=2 if _flags().fuse_residual_requant else 1)
    if tr:
        tr.fx("rewrites", gm, "sink_permute {}, inline_mha {}, split_to_narrow {}, "
                              "select_to_narrow {} (ours, on the fx graph)".format(*moved))
        tr.ops("graph ops", fg, "one graph-API op per fx node; a wildcard is never fused into a partition")
    by_name = {n.name: n for n in gm.graph.nodes}
    sup, cps = [], []
    for p in fg.parts:             # compile first: a partition that fails to compile stays torch code
        if not p.supported:
            continue
        if all(fg.ops[o][0] == "reshape" for o in p.ops):
            continue                   # a view in torch (zero copy); as a partition it would be a copy
        try:
            cps.append(p.compile())
            sup.append(p)
        except RuntimeError as exc:
            fg.compile_failed[p.info] = str(exc)
    if tr:
        tr.parts("partitions", fg, sup, cps, "the library's partitions: supported = the fused primitive created")
    groups = [_group_nodes(fg, p, by_name) for p in sup]
    bounds = [_boundary(nodes) for nodes in groups]   # before fusing renames the inputs
    fused = fuse_by_partitions(gm, groups)
    mods = {}
    for k, (p, cp, (ins, outs)) in enumerate(zip(sup, cps, bounds)):
        name = f"fused_{k}"
        mods[name] = _PartModule(fg, p, cp, ins, outs)
        setattr(fused, name, mods[name])
    fused.recompile()
    fused.aiednn_graph, fused.aiednn_parts = fg, mods
    if tr:
        tr.fused("executed", fused, mods, "what torch runs: a node per compiled partition, the rest torch")
        print(f"[aiednn] stage graphs: {tr.write(_flags().dump)}")
    aiednn_backend.last = fused
    aiednn_backend.history.append(fused)    # every graph Dynamo compiled (graph breaks give several)
    return fused.forward


def aiednn_backend(gm, example_inputs):
    """the Dynamo backend: partition, compile the supported regions, return the stitched forward.
    The graph API decides the partitions when it is in the tree and ``graph`` is not False (f32
    graphs); otherwise the interim dnn_torch.lower regions."""
    import torch
    _freeze_params(gm, example_inputs)
    fl = _flags()
    if fl.graph is not False and fl.mode == "f32" and _graph_api():
        return _graph_backend(gm, example_inputs)
    if fl.graph:
        raise AiegraphError("dnn_torch_backend", "graph=True needs the partition graph API and mode='f32'")
    names = {n.name for n in gm.graph.nodes}
    unsupported: set = set()
    for _ in range(len(names) + 1):   # each failing lowering hands one node back to torch
        fused = _partition(gm, unsupported)
        inputs = _region_inputs(fused, example_inputs)
        regions, failed = {}, None
        for target, (args, const) in inputs.items():
            acts = [i for i, a in enumerate(args) if not const[i] and isinstance(a, torch.Tensor)]
            if len(acts) != 1 or not all(isinstance(a, torch.Tensor) for a in args):
                continue                  # interim: one activation per region
            try:
                regions[target] = _Region(getattr(fused, target), args, acts[0])
            except AiegraphError as exc:
                m = _NODE.search(str(exc))
                node = m and (m.group(1) or m.group(2))
                if node in names and node not in unsupported:
                    failed = node
                    break                 # re-partition without it
        if failed is not None:
            unsupported.add(failed)
            continue
        for target, region in regions.items():
            setattr(fused, target, _RegionModule(region))
        fused.recompile()
        fused.aiednn_regions = regions    # introspection: fused_* name -> _Region
        fused.aiednn_unsupported = sorted(unsupported)
        aiednn_backend.last = fused
        aiednn_backend.history.append(fused)
        return fused.forward
    raise AiegraphError("dnn_torch_backend", "partitioning did not converge")


class _RegionModule(torch.nn.Module):
    """stands in for a fused_* submodule (fx calls it by attribute)"""

    def __init__(self, region):
        super().__init__()
        self.region = region

    def forward(self, *args):
        return self.region(*args)


def register():
    """make backend="aiednn" known to torch.compile (done by importing aiegraph.backends.aiednn)"""
    from torch._dynamo import list_backends, register_backend
    if "aiednn" not in list_backends(exclude_tags=()):
        register_backend(compiler_fn=aiednn_backend, name="aiednn")


aiednn_backend.history = []
register()
