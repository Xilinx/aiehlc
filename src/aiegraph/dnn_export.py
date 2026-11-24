###############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
"""torch.export -> a board program, through the offline model compiler (flow 4).

``torch.export.export(model, (x,))`` gives an ExportedProgram: an ATen graph with the parameters lifted. The
model compiler (aiegraph.dnn_torch -> dnn_mlir -> emit_app -> dnn_build) lowers torch-level fx graphs: module
calls (nn.Conv2d, nn.Linear, the norms) and torch / functional calls. ``to_fx`` translates the one into the
other: an op that carries parameters becomes the module it came from, rebuilt from the program's own tensors
(aten.convolution -> nn.Conv2d, addmm / mm of a transposed constant -> nn.Linear, the batch / layer norm ops ->
nn.BatchNorm2d / nn.LayerNorm), every other op the torch function of the same name, and a node that depends on
no input is evaluated once and kept as a constant. Everything after that is flow 4 unchanged: Conv+BN folding,
int8 calibration on the example input, the MLIR pipeline, the C app, the aiehlc build.

    ep = torch.export.export(model, (x,))
    rep, g, ref = aiegraph.dnn_export.compile_exported(ep, x, target="vek385")   # the board program (ELF)
    aiegraph.dnn_export.run(rep, board="<host>")                                  # run it on any VEK385

    # Host-owned main: emit="model" writes <name>_model.h (create/run/destroy) + a smoke host for the build.
    rep, g, ref = aiegraph.dnn_export.compile_exported(ep, x, emit="model", target="x86")
"""

from __future__ import annotations

from contextlib import contextmanager
import operator
from pathlib import Path

from .errors import AiegraphError

__all__ = ["prepare_for_export", "to_fx", "compile_exported", "run"]


def prepare_for_export(model):
    """*model* with every nn.MultiheadAttention swapped for the same math in modules the model compiler fuses
    (packed q / k / v Linear, head split, sdpa, out Linear; batch-first). Export the result: exported as is,
    MultiheadAttention decomposes sequence-first with selects and transposes around sdpa (vit_b_16: 284
    primitives instead of 116), which still compiles, just with the copies on the host."""
    from . import dnn_torch as dt
    return dt._decompose_mha(model.eval())


class _Multi:
    """the tuple an aten op returns (norm / pool / attention): getitem picks the element"""
    def __init__(self, items):
        self.items = list(items)


class _ConstT:
    """aten.t of a constant weight, kept until an addmm / mm turns it into an nn.Linear"""
    def __init__(self, node, tensor):
        self.node, self.tensor = node, tensor


def _fn_table():
    import torch
    import torch.nn.functional as F
    a = torch.ops.aten
    same = {a.relu.default: torch.relu, a.sigmoid.default: torch.sigmoid, a.tanh.default: torch.tanh,
            a.silu.default: F.silu, a.hardswish.default: F.hardswish, a.hardsigmoid.default: F.hardsigmoid,
            a.neg.default: torch.neg, a.rsqrt.default: torch.rsqrt, a.exp.default: torch.exp, a.sqrt.default: torch.sqrt,
            a.abs.default: torch.abs, a.erf.default: torch.erf, a.pow.Tensor_Scalar: torch.pow,
            a.add.Tensor: operator.add, a.sub.Tensor: operator.sub, a.mul.Tensor: operator.mul,
            a.div.Tensor: operator.truediv, a.add.Scalar: operator.add, a.mul.Scalar: operator.mul,
            a.maximum.default: torch.maximum, a.minimum.default: torch.minimum,
            a.permute.default: torch.permute, a.transpose.int: torch.transpose, a.unsqueeze.default: torch.unsqueeze,
            a.squeeze.dim: torch.squeeze, a.cat.default: torch.cat, a.split.Tensor: torch.split,
            a.split_with_sizes.default: torch.split, a.bmm.default: torch.bmm, a.matmul.default: torch.matmul,
            a.mean.dim: torch.mean, a.adaptive_avg_pool2d.default: F.adaptive_avg_pool2d,
            a.avg_pool2d.default: F.avg_pool2d, a.leaky_relu.default: F.leaky_relu}
    reshape = {a.view.default, a._unsafe_view.default, a.reshape.default}
    ident = {a.clone.default, a.alias.default, a.detach.default, a.contiguous.default, a.dropout.default}
    return same, reshape, ident


def _is_dynamic(v) -> bool:
    import torch
    if isinstance(v, torch.fx.Node):
        return True
    if isinstance(v, (_Multi, _ConstT)):
        return True
    if isinstance(v, (list, tuple)):
        return any(_is_dynamic(x) for x in v)
    return False


class _Normalized:
    """an aten call with its positional arguments completed from keywords and schema defaults: a graph
    deserialized by torch.export.load spells max_pool2d's stride / padding as keywords where a fresh export
    passes them positionally, and reading only n.args mistranslated the pool (stride = kernel, no padding)"""

    def __init__(self, n):
        self.node, self.name, self.target, self.meta = n, n.name, n.target, n.meta
        schema = getattr(n.target, "_schema", None)
        if schema is None:
            self.args, self.kwargs = tuple(n.args), dict(n.kwargs)
            return
        args, kwargs = list(n.args), {}
        for i, arg in enumerate(schema.arguments):
            if arg.kwarg_only:
                if arg.name in n.kwargs:
                    kwargs[arg.name] = n.kwargs[arg.name]
                elif arg.has_default_value():
                    kwargs[arg.name] = arg.default_value
            elif i >= len(args):
                if arg.name in n.kwargs:
                    args.append(n.kwargs[arg.name])
                elif arg.has_default_value():
                    args.append(arg.default_value)
                else:
                    break
        self.args, self.kwargs = tuple(args), kwargs


#: namespaces of PyTorch's own operators; any other OpOverload is a user / library custom op
CORE_NAMESPACES = ("aten", "prims", "quantized_decomposed", "_quantized", "quantized")


def is_custom_op(t) -> bool:
    """a framework custom op (torch.library) the exported graph keeps opaque: a call of torch.ops.<ns>.<op>
    outside PyTorch's own namespaces"""
    import torch
    return isinstance(t, torch._ops.OpOverload) and t.namespace not in CORE_NAMESPACES


def _op_body(op):
    """the Python body of a torch.library.custom_op (newer torch keeps it), else None"""
    try:
        from torch._library.custom_ops import OPDEFS
    except ImportError:
        return None
    d = OPDEFS.get(op.name() if hasattr(op, "name") else str(op))
    return getattr(d, "_init_fn", None) if d is not None else None


@contextmanager
def inline_python_ops(bodies=None):
    """while exporting, run Python-implemented custom ops as their body so torch.export inlines its ATen IR: the op
    then lowers like any other layer (to the AIE where it can) instead of staying one opaque node on the host.
    *bodies*: {op or "ns::op": python function written in torch ops}; ops made with torch.library.custom_op are
    found by themselves. Each body is registered as the op's CompositeImplicitAutograd kernel for the duration
    only. A body registered for a backend key ("CPU") cannot be read back from the dispatcher: pass it here.

        with dnn_export.inline_python_ops({"mylib::softclip": lambda x: torch.tanh(x) * 3}):
            ep = torch.export.export(model, (x,))"""
    import torch
    libs = []
    try:
        for op, fn in (bodies or {}).items():
            name = op if isinstance(op, str) else op.name()
            ns, opname = name.split("::")
            lib = torch.library.Library(ns, "IMPL")
            lib.impl(opname.split(".")[0], fn, "CompositeImplicitAutograd")
            libs.append(lib)
        yield
    finally:
        for lib in libs:                       # deregister: the op is opaque again after export
            lib._destroy()


def export_inlined(model, args, bodies=None):
    """torch.export.export with every Python-implemented custom op inlined (inline_python_ops): the bodies given
    in *bodies* ({op or "ns::op": fn}) and those of torch.library.custom_op ops. Ops with no Python body stay
    one opaque node and run on the host (aiednn_custom: a C implementation, compile_exported(custom_sources=))."""
    import torch
    ep = torch.export.export(model, tuple(args))
    found = dict(bodies or {})
    for n in ep.graph.nodes:
        if n.op == "call_function" and is_custom_op(n.target):
            name = n.target.name().split(".")[0]
            if name not in found and (fn := _op_body(n.target)) is not None:
                found[name] = fn
    if not found:
        return ep
    with inline_python_ops(found):
        return torch.export.export(model, tuple(args))


class _Translator:
    """one pass over ep.module()'s graph building the torch-level graph (call() is one node)"""

    def __init__(self, ep):
        import torch
        self.torch = torch
        self.src = ep.module()
        self.root = torch.nn.Module()
        self.graph = torch.fx.Graph()
        self.env = {}
        self.n_const = 0
        self.same, self.reshape, self.ident = _fn_table()

    # ── values ──
    def const(self, t):
        """a constant tensor as a get_attr node on the new root"""
        name = f"_c{self.n_const}"
        self.n_const += 1
        self.root.register_buffer(name, t.detach().clone())
        return self.graph.get_attr(name)

    def module(self, prefix, m, *args):
        name = f"{prefix}{len(list(self.root.children()))}"
        self.root.add_module(name, m.eval())
        return self.graph.call_module(name, tuple(args))

    def val(self, v):
        """an argument for the new graph: nodes mapped, _ConstT materialized"""
        import torch
        if isinstance(v, torch.fx.Node):
            r = self.env[v]
            if isinstance(r, _ConstT):
                r = self.graph.call_function(torch.transpose, (self.const(r.tensor), 0, 1))
                self.env[v] = r
            elif isinstance(r, torch.Tensor):     # a folded constant used by a dynamic op: a buffer on the root
                r = self.const(r)
                self.env[v] = r
            return r
        if isinstance(v, (list, tuple)):
            return type(v)(self.val(x) for x in v)
        return v

    def tensor_of(self, v):
        """the constant tensor behind a mapped value (None if it depends on the input)"""
        import torch
        r = self.env.get(v) if isinstance(v, torch.fx.Node) else v
        return r if isinstance(r, torch.Tensor) else None

    # ── ops ──
    def conv(self, n):
        torch = self.torch
        x, w, b, stride, pad, dil, transposed, out_pad, groups = n.args
        W, B = self.tensor_of(w), self.tensor_of(b) if b is not None else None
        if W is None or (b is not None and B is None) or W.dim() != 4:
            raise AiegraphError("dnn_export", f"{n.name}: convolution needs constant 2-D weights")
        cls = torch.nn.ConvTranspose2d if transposed else torch.nn.Conv2d
        cin = W.shape[0] if transposed else W.shape[1] * groups
        cout = W.shape[1] * groups if transposed else W.shape[0]
        kw = dict(stride=tuple(stride), padding=tuple(pad), dilation=tuple(dil), groups=groups, bias=B is not None)
        if transposed:
            kw["output_padding"] = tuple(out_pad)
        m = cls(cin, cout, tuple(W.shape[2:]), **kw)
        m.weight.data = W.clone()
        if B is not None:
            m.bias.data = B.clone()
        return self.module("conv", m, self.val(x))

    def batch_norm(self, n):
        torch = self.torch
        x, w, b, rm, rv, _momentum, eps = n.args
        RM, RV = self.tensor_of(rm), self.tensor_of(rv)
        m = torch.nn.BatchNorm2d(RM.shape[0], eps=eps, affine=w is not None)
        m.running_mean.data, m.running_var.data = RM.clone(), RV.clone()
        if w is not None:
            m.weight.data, m.bias.data = self.tensor_of(w).clone(), self.tensor_of(b).clone()
        return _Multi([self.module("bn", m, self.val(x)), None, None])

    def layer_norm(self, n):
        torch = self.torch
        x, shape, w, b, eps = n.args
        m = torch.nn.LayerNorm(list(shape), eps=eps, elementwise_affine=w is not None)
        if w is not None:
            m.weight.data = self.tensor_of(w).clone()
            if b is not None:
                m.bias.data = self.tensor_of(b).clone()
            else:
                m.bias = None
        return _Multi([self.module("ln", m, self.val(x)), None, None])

    def linear(self, n, x, wt, b):
        """x @ t(W) (+ b) with W constant -> nn.Linear, else None"""
        torch = self.torch
        r = self.env.get(wt) if isinstance(wt, torch.fx.Node) else None
        if not isinstance(r, _ConstT) or (b is not None and self.tensor_of(b) is None):
            return None
        W = r.tensor
        m = torch.nn.Linear(W.shape[1], W.shape[0], bias=b is not None)
        m.weight.data = W.clone()
        if b is not None:
            m.bias.data = self.tensor_of(b).reshape(-1).clone()
        return self.module("fc", m, self.val(x))

    def sdpa(self, n):
        F = self.torch.nn.functional
        q, k, v = n.args[:3]
        kw = dict(dropout_p=n.args[3] if len(n.args) > 3 else n.kwargs.get("dropout_p", 0.0),
                  is_causal=n.args[4] if len(n.args) > 4 else n.kwargs.get("is_causal", False))
        if n.kwargs.get("scale") is not None:
            kw["scale"] = n.kwargs["scale"]
        if n.kwargs.get("attn_mask") is not None:
            kw["attn_mask"] = self.val(n.kwargs["attn_mask"])
        y = self.graph.call_function(F.scaled_dot_product_attention, (self.val(q), self.val(k), self.val(v)), kw)
        return _Multi([y] + [None] * 8)

    def nearest(self, n):
        """x[:, :, ih, iw] with constant index grids floor(i * in / out): the nearest upsample it decomposed from"""
        torch = self.torch
        x, idx = n.args
        H, W = n.args[0].meta["val"].shape[-2:]
        ih = self.tensor_of(idx[2]) if len(idx) == 4 and idx[0] is None and idx[1] is None else None
        iw = self.tensor_of(idx[3]) if ih is not None else None
        if ih is None or iw is None:
            raise AiegraphError("dnn_export", f"{n.name}: _unsafe_index other than a nearest upsample")
        ih, iw = ih.reshape(-1), iw.reshape(-1)
        Ho, Wo = ih.numel(), iw.numel()
        if not (torch.equal(ih, torch.arange(Ho) * H // Ho) and torch.equal(iw, torch.arange(Wo) * W // Wo)):
            raise AiegraphError("dnn_export", f"{n.name}: _unsafe_index grid is not a nearest upsample")
        return self.graph.call_function(torch.nn.functional.interpolate, (self.val(x),),
                                        dict(size=(Ho, Wo), mode="nearest"))

    def call(self, n):  # noqa: C901 -- one branch per aten op family
        torch, a, g = self.torch, self.torch.ops.aten, self.graph
        n = _Normalized(n)
        t = n.target
        if t is operator.getitem:
            src = self.env[n.args[0]]
            return src.items[n.args[1]] if isinstance(src, _Multi) else g.call_function(operator.getitem, self.val(n.args))
        if not _is_dynamic([self.env.get(x) if isinstance(x, torch.fx.Node) else x for x in n.args]) and \
                t not in (a.t.default,):      # depends on no input: evaluate once (yolov8n's anchor grid)
            args = [self.env.get(x) if isinstance(x, torch.fx.Node) else x for x in n.args]
            return t(*args, **n.kwargs)
        if t is a.convolution.default:
            return self.conv(n)
        if t is a._native_batch_norm_legit_no_training.default:
            return self.batch_norm(n)
        if t is a.native_layer_norm.default:
            return self.layer_norm(n)
        if t is a.t.default:
            W = self.tensor_of(n.args[0])
            return _ConstT(n, W) if W is not None else g.call_function(torch.transpose, (self.val(n.args[0]), 0, 1))
        if t is a.addmm.default:
            y = self.linear(n, n.args[1], n.args[2], n.args[0])
            return y if y is not None else g.call_function(operator.add, (
                self.val(n.args[0]), g.call_function(torch.matmul, (self.val(n.args[1]), self.val(n.args[2])))))
        if t is a.mm.default:
            y = self.linear(n, n.args[0], n.args[1], None)
            return y if y is not None else g.call_function(torch.matmul, self.val(n.args))
        if t in (a._scaled_dot_product_flash_attention_for_cpu.default, a.scaled_dot_product_attention.default):
            return self.sdpa(n)
        if t is a.max_pool2d_with_indices.default:
            F = torch.nn.functional
            x, k, s, p, d, ceil = (list(n.args) + [None] * 6)[:6]
            y = g.call_function(F.max_pool2d, (self.val(x), k), dict(stride=s or k, padding=p or 0, dilation=d or 1,
                                                                       ceil_mode=bool(ceil)))
            return _Multi([y, None])
        if t is a._unsafe_index.Tensor:
            return self.nearest(n)
        if t is a._softmax.default:
            return g.call_function(torch.softmax, (self.val(n.args[0]), n.args[1]))
        if t is a.gelu.default:
            return g.call_function(torch.nn.functional.gelu, (self.val(n.args[0]),),
                                   dict(approximate=n.kwargs.get("approximate", "none")))
        if t is a.hardtanh.default:
            return g.call_function(torch.nn.functional.hardtanh, (self.val(n.args[0]),) + tuple(n.args[1:3]))
        if t is a.expand.default:
            return g.call_function(torch.Tensor.expand, (self.val(n.args[0]), list(n.args[1])))
        if t is a.slice.Tensor:
            x, dim, start, end, step = list(n.args) + [0, None, None, 1][len(n.args) - 1:]   # aten defaults
            start = 0 if start is None else start
            size = n.args[0].meta["val"].shape[dim]
            end = size if end is None or end > size else end
            idx = (slice(None),) * dim + (slice(start, end, step),)
            return g.call_function(operator.getitem, (self.val(x), idx))
        if t is a.select.int:
            idx = (slice(None),) * n.args[1] + (n.args[2],)
            return g.call_function(operator.getitem, (self.val(n.args[0]), idx))
        if t in self.reshape:
            return g.call_function(torch.reshape, (self.val(n.args[0]), list(n.args[1])))
        if t in self.ident:
            return self.val(n.args[0])
        if t in self.same:
            # positional args are normalized; of the keywords, only the call's own keyword-only ones (schema defaults
            # such as add's alpha=1 are not arguments of operator.add & co)
            kw = {k: v for k, v in n.node.kwargs.items() if k in n.kwargs}
            return g.call_function(self.same[t], self.val(tuple(n.args)), self.val(kw))
        if is_custom_op(t):   # a framework custom op: kept as itself; the model compiler runs it on the host
            return g.call_function(t, self.val(tuple(n.args)), self.val({k: v for k, v in n.node.kwargs.items()}))
        raise AiegraphError("dnn_export", f"{n.name}: {t} has no torch-level counterpart in dnn_export yet",
                            hint="add it to dnn_export._Translator.call / _fn_table")

    def run(self):
        torch = self.torch
        params = dict(self.src.named_parameters())
        params.update(dict(self.src.named_buffers()))
        for n in self.src.graph.nodes:
            if n.op == "placeholder":
                self.env[n] = self.graph.placeholder(n.name)
            elif n.op == "get_attr":
                obj = self.src
                for part in n.target.split("."):
                    obj = getattr(obj, part)
                self.env[n] = obj.detach() if isinstance(obj, torch.Tensor) else obj
            elif n.op == "call_function":
                self.env[n] = self.call(n)
            elif n.op == "output":
                outs = [o for o in torch.utils._pytree.tree_flatten(n.args[0])[0] if isinstance(o, torch.fx.Node)]
                self.graph.output(self.val(outs[0]))
            else:
                raise AiegraphError("dnn_export", f"{n.name}: unexpected {n.op} in an exported graph")
        gm = torch.fx.GraphModule(self.root, self.graph)
        gm.graph.eliminate_dead_code()
        gm.recompile()
        return gm


def _shape(n):
    tm = n.meta.get("tensor_meta")
    return tuple(tm.shape) if tm is not None and hasattr(tm, "shape") else None


def _fold_linear_views(gm) -> int:
    """reshape(x, [prod(lead), K]) -> Linear -> reshape(y, lead + [N]) back to Linear(x): export flattens every
    Linear on a 3-D input, and the views between a GEMM and its add / activation / quantize keep the model
    compiler from fusing them (gpt2: 36 slices, 48 reorders, 36 epilogue ops). Returns the count."""
    import torch
    n_fold = 0
    for n in list(gm.graph.nodes):
        if n.op != "call_module" or not isinstance(gm.get_submodule(n.target), torch.nn.Linear):
            continue
        r_in = n.args[0]
        if not (isinstance(r_in, torch.fx.Node) and r_in.target is torch.reshape) or len(n.users) != 1:
            continue
        r_out = next(iter(n.users))
        x = r_in.args[0]
        xs, ys = _shape(x), _shape(r_out)
        if r_out.target is not torch.reshape or xs is None or ys is None or xs[:-1] != ys[:-1]:
            continue
        n.args = (x,)
        r_out.replace_all_uses_with(n)
        gm.graph.erase_node(r_out)
        if not r_in.users:
            gm.graph.erase_node(r_in)
        n_fold += 1
    return n_fold


def _fold_gqa_expand(gm) -> int:
    """reshape(expand(unsqueeze(x, d)), merge of dims d-1, d) -> repeat_interleave(x, rep, d-1): the K / V
    head repeat of grouped-query attention as export writes it (the model compiler folds repeat_interleave
    into a GQA sdpa). Returns the count."""
    import torch
    n_fold = 0
    for n in list(gm.graph.nodes):
        if n.target is not torch.reshape:
            continue
        e = n.args[0]
        if not (isinstance(e, torch.fx.Node) and e.target is torch.Tensor.expand):
            continue
        u = e.args[0]
        if not (isinstance(u, torch.fx.Node) and u.target is torch.unsqueeze):
            continue
        x, d = u.args[0], u.args[1]
        xs, es, ns = _shape(x), _shape(e), _shape(n)
        if None in (xs, es, ns) or d < 1 or len(ns) != len(xs) or ns[d - 1] != es[d - 1] * es[d]:
            continue
        with gm.graph.inserting_before(n):
            r = gm.graph.call_function(torch.repeat_interleave, (x, int(es[d])), dict(dim=d - 1))
        r.meta = dict(n.meta)
        n.replace_all_uses_with(r)
        for dead in (n, e, u):
            if not dead.users:
                gm.graph.erase_node(dead)
        n_fold += 1
    return n_fold


def to_fx(ep, example_inputs=None):
    """ExportedProgram -> a torch-level fx GraphModule the model compiler lowers (see the module docstring)"""
    import torch
    from torch.fx.passes.shape_prop import ShapeProp
    gm = _Translator(ep).run()
    if example_inputs is None:   # the example the program was exported with (its placeholders' fake values)
        example_inputs = [torch.zeros(n.meta["val"].shape, dtype=n.meta["val"].dtype)
                          for n in ep.graph.nodes if n.op == "placeholder" and n.name in ep.graph_signature.user_inputs]
    ShapeProp(gm).propagate(*example_inputs)
    if _fold_linear_views(gm) + _fold_gqa_expand(gm):
        gm.graph.eliminate_dead_code()
        gm.recompile()
    return gm


def compile_exported(ep, x, *, mode: str = "int8", name: str = "model", target: str = "x86", out=None,
                     vitis_settings: str = None, mlir: bool = True, prefer: str = "fastest", mesh=None,
                     top1_range=None, eager=None, custom_sources=(), emit: str = "app",
                     eager_mesh: str = "4x4", synth_weights: bool = False):
    """An exported program -> the program for *target* (x86 | vek280 | vek385): flow 4 on to_fx(ep), built, not
    run. Returns (DnnBuild, AGraph, reference); run it with run(). *mode* f32 | int8 (calibrated on *x*);
    *top1_range*: a detector's class-score window (zoo's model.top1_range). *eager*: None builds the
    fixed-megakernel program; ``"plan"`` eager mode on instances generated for this model's layer shapes
    (dnn_eager.plan); ``"library"`` eager mode on the premade template library (dnn_eager.LIBRARY), no
    per-model kernels. *eager_mesh*: the instances' mesh, ``"RxC"`` with R in 4 / 6 / 8 and C = 4 (8x4 with
    warm launch is the fastest on VEK280). *custom_sources*: C files implementing the program's framework custom ops
    (dnn_torch.custom_c_symbol), run on the host CPU.
    *emit*: ``"app"`` (default) writes the full benchmark ``main``; ``"model"`` writes a session library
    (``<name>_model.h`` / ``.cc``: create / run / destroy) plus a thin smoke host for the build check —
    replace the smoke with your own host that includes the model header.
    *synth_weights*: the app generates its large int8 weights on the device (dnn_torch.synthesize_weights)
    instead of carrying them, so a board ELF downloads in seconds; the reference is then the aiednn graph run
    on the workstation with the same weights, so this checks the build and its timing, not accuracy."""
    from . import dnn_torch as dt
    from ._env import repo_root
    from .dnn import dnn_build
    if emit not in ("app", "model"):
        raise AiegraphError("dnn_export", f"emit={emit!r}", hint="'app' or 'model'")
    if synth_weights and emit != "app":
        raise AiegraphError("dnn_export", "synth_weights with emit='model'", hint="synthetic weights are an app option")
    gm = to_fx(ep, [x])
    if top1_range is not None:
        gm.top1_range = top1_range
    out = Path(out or repo_root() / "worklocal" / "aiednn" / "export" / (
        f"{name}_{mode}" + (f"_eager_{eager}" + (f"_{eager_mesh}" if eager_mesh != "4x4" else "") if eager else "") + (f"_emit_{emit}" if emit != "app" else "")
        + ("_synth" if synth_weights else "")))
    g, ref, yf = dt.references(gm, x, mode, name, mlir=mlir)
    float_top1 = dt._top1(yf, g.top1_range)[0]
    if synth_weights:                  # golden = the same graph with the same synthetic weights, on the workstation
        import numpy as np

        from .dnn_ctypes import AiednnLib
        moved = dt.synthesize_weights(g)
        lib = AiednnLib.build(out / "x86lib")
        ref = np.asarray(dt.run_ctypes(g, lib, x.numpy())[0], np.float32).reshape(ref.shape)
        float_top1 = -1
        print(f"synthetic weights: {len(g.synth)} int8 tensors, {moved / 1e6:.1f} MB generated on the device", flush=True)
    if emit == "model":
        dt.emit_model(g, out, prefer=prefer, custom_sources=custom_sources)
        app = dt.emit_model_smoke(g, out, ref, dt.TOL[mode], dt._top1(yf, g.top1_range)[0],
                                  input_data=g.tensors[g.root(g.input)].data)
    else:
        app = dt.emit_app(g, out, ref, dt.TOL[mode], float_top1, prefer=prefer,
                          custom_sources=custom_sources)
    kw = {}
    if eager:                          # every layer on a template instance, its kernel loaded right before it runs
        from . import dnn_eager
        if eager not in ("plan", "library"):
            raise AiegraphError("dnn_export", f"eager={eager!r}", hint="'plan' or 'library'")
        rows, _, cols = eager_mesh.partition("x")
        if not rows.isdigit() or int(rows) not in dnn_eager.MESH_ROWS or cols != "4":
            raise AiegraphError("dnn_export", f"eager_mesh={eager_mesh!r}",
                                hint=f"RxC with R in {dnn_eager.MESH_ROWS} and C = 4")
        rows = int(rows)
        plan = dnn_eager.library_plan(name, rows) if eager == "library" else dnn_eager.plan(g, name, rows)
        print(plan.summary(), flush=True)
        kw = dict(mode="eager", eager=plan)
    rep = dnn_build(app, target=target, out=out / target, run=False, vitis_settings=vitis_settings, mesh=mesh, **kw)
    return rep, g, ref


def run(rep, board: str = None, timeout: int = 1800):
    """Run a compiled program: an x86 one here, a board one on *board* (any board of its target: the ELF does
    not depend on which). Sets rep.run_ok / rep.run_log; the app prints MODEL ... and PASS / FAIL against the
    PyTorch reference it carries."""
    from .dnn import run_app
    if rep.target != "x86" and not board:
        raise AiegraphError("dnn_export", f"a {rep.target} program runs on a board", hint="run(rep, board=<host>)")
    return run_app(rep, board=board, timeout=timeout)
