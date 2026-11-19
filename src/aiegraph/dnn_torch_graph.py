"""PyTorch fx graph -> aiednn graph API (aiegraph.dnn_graph, include/aiednn_graph.h).

One graph op per fx node, named by the node: a module / function / method aiednn has a graph-op kind
for maps onto it (constants -- weights, biases, BN statistics, python scalars -- become constant
logical tensors), everything else becomes a ``wildcard`` op so the partitioner sees the whole graph and
never fuses across it. Nodes whose value is not a tensor (``x.size()``, ``getitem`` of a python tuple)
stay out of the graph; the tensors they read are marked graph outputs. The library then decides
support: ``FxGraph.parts`` are the graph API's partitions, a supported one compiles to one aiednn
primitive.

    fg = FxGraph(lib, gm, example_inputs)      # gm: fx GraphModule (Dynamo's, or symbolic_trace)
    fg.report()                                # partitions, implementation, unsupported
    fg.ops                                     # node name -> (kind, inputs, attrs): the mapping itself

Used by ``torch.compile(model, backend="aiednn")`` (aiegraph.dnn_torch_backend); the mapping is the
fx side of the graph-vs-MLIR fusion check of doc/design/aiednn_framework_api.md section 1.
"""
from __future__ import annotations

import math
import operator

import numpy as np

from .dnn_torch import _ALIAS_FN, _ELTWISE_FN, _ELTWISE_MOD, _IDENTITY_MOD

#: dnn_torch eltwise alg -> the dedicated graph-op kind (fuses as a post-op either way)
_KIND = {"relu": "relu", "clip": "clamp", "logistic": "sigmoid", "tanh": "tanh", "gelu": "gelu"}
_BIN = {"add": "add", "iadd": "add", "mul": "multiply", "imul": "multiply", "sub": "binary_sub",
        "isub": "binary_sub", "truediv": "binary_div", "div": "binary_div", "maximum": "binary_max",
        "minimum": "binary_min"}
_TORCH_DT = {"torch.float32": "f32", "torch.int8": "s8", "torch.uint8": "u8", "torch.int32": "s32",
             "torch.bool": "u8"}   # bool: one byte, nonzero = true (select's cond)


def _name(target) -> str:
    """the op name of a call target: torch.relu -> relu, aten.conv2d.default -> conv2d,
    quantized_decomposed.quantize_per_tensor.default -> quantize_per_tensor"""
    schema = getattr(target, "_schema", None)    # torch._ops.OpOverload: "aten::conv2d"
    if schema is not None:
        return schema.name.split("::")[-1]
    return getattr(target, "__name__", str(target)).split(".")[-1]


def _pair(v):
    return tuple(v) if isinstance(v, (tuple, list)) else (v, v)


class FxGraph:
    def __init__(self, lib, gm, example_inputs, fuse: int = 1):   # graph API fusion level (aiednn_graph_set_fusion)
        import torch
        from torch.fx.passes.shape_prop import ShapeProp

        from .dnn_graph import Graph
        self.gm, self.torch = gm, torch
        vals = self.pyvals = {}   # node name -> its python value (shape arithmetic: x.size(-1) // 2)

        class _Prop(ShapeProp):
            def run_node(self, n):
                r = super().run_node(n)
                if isinstance(r, (int, float, bool)) or (isinstance(r, (tuple, list)) and
                                                         all(isinstance(v, int) for v in r)):
                    vals[n.name] = r
                return r
        _Prop(gm).propagate(*example_inputs)
        self.mods = dict(gm.named_modules())
        self.graph = Graph(lib, fuse=fuse)
        self.consts = {}     # constant tensor name -> numpy (fed to every execute)
        self.ops = {}        # node name -> (kind, input tensor names, attrs)
        self.wildcards = {}  # node name -> why
        self.alias = {}      # node name -> the graph tensor it stands for (a GQA K / V head repeat folded into sdpa)
        self.compile_failed = {}  # partition info -> error (dnn_torch_backend: those stay torch code)
        marks = []
        for node in gm.graph.nodes:
            meta = node.meta.get("tensor_meta")
            is_t = meta is not None and hasattr(meta, "shape")
            if node.op == "placeholder":
                if is_t and node.users:
                    self._tensor(node.name, meta)
            elif node.op == "get_attr":
                if is_t:
                    self._const(node.name, self._attr(node.target))
            elif node.op == "output":
                marks += [a.name for a in self._tensor_args(node.args)]
            elif not is_t or str(meta.dtype) not in _TORCH_DT:
                marks += [a.name for a in self._tensor_args((node.args, node.kwargs))]
            else:
                self._add(node, meta)
        for m in dict.fromkeys(marks):
            if m in self.graph.tensors:
                self.graph.mark_output(m)
        self.parts = self.graph.compile_all()

    # ── tensors ──────────────────────────────────────────────────────────
    def _tensor(self, name, meta, const=False):
        self.graph.tensor(name, tuple(int(d) for d in meta.shape), _TORCH_DT.get(str(meta.dtype), "f32"), const)

    def _const(self, name, value, rank=None) -> str:
        arr = value.detach().cpu().numpy() if isinstance(value, self.torch.Tensor) else np.asarray(value, np.float32)
        arr = np.ascontiguousarray(arr if arr.dtype != np.float64 else arr.astype(np.float32))
        if rank is not None and arr.ndim < rank:
            arr = arr.reshape((1,) * (rank - arr.ndim) + arr.shape)
        dt = {np.float32: "f32", np.int8: "s8", np.uint8: "u8", np.int32: "s32"}.get(arr.dtype.type, "f32")
        self.consts[name] = arr.astype(np.float32) if dt == "f32" else arr
        self.graph.tensor(name, arr.shape, dt, const=True)
        return name

    def _attr(self, target):
        v = self.gm
        for p in target.split("."):
            v = getattr(v, p)
        return v

    def _tensor_args(self, args) -> list:
        import torch.fx as fx
        out = []
        def walk(a):
            if isinstance(a, fx.Node):
                if a.name in self.graph.tensors:
                    out.append(a)
            elif isinstance(a, (list, tuple)):
                for x in a:
                    walk(x)
            elif isinstance(a, dict):
                for x in a.values():
                    walk(x)
        walk(args)
        return out

    # ── ops ──────────────────────────────────────────────────────────────
    def _add(self, node, meta):
        src = self._gqa_repeat(node)
        if src is not None:   # sdpa reads the Hk heads itself: no op, no tensor
            self.alias[node.name] = src
            return
        self._tensor(node.name, meta)
        if _name(node.target) in ("slice", "select") and node.op == "call_function" and "slice" in self.graph_kinds() \
                and self._aten_slice(node, meta):   # aten.slice.Tensor / aten.select.int (PT2E graphs)
            return
        if _name(node.target) == "narrow" and "slice" in self.graph_kinds():   # the backend's split pieces
            x, dim, start = node.args[0], node.args[1], self._pv(node.args[2])
            if hasattr(x, "name") and x.name in self.graph.tensors:
                starts = [0] * len(meta.shape)
                starts[dim % len(meta.shape)] = int(start)
                self.ops[node.name] = ("slice", [x.name], {"starts": tuple(starts)})
                self.graph.add("slice", [x.name], node.name, name=node.name, starts=tuple(starts))
                return
        if node.op == "call_function" and node.target is operator.getitem and "slice" in self.graph_kinds():
            try:
                if self._getitem(node, meta):
                    return
            except (KeyError, TypeError, ValueError, AttributeError, IndexError):
                pass
        try:
            kind, ins, kw = self._map(node)
        except (KeyError, TypeError, ValueError, AttributeError, IndexError) as exc:
            kind, ins, kw = None, [], {"why": f"{type(exc).__name__}: {exc}"}
        if kind is None:
            self.wildcards[node.name] = kw.get("why", self._what(node))
            kind, ins, kw = "wildcard", [a.name for a in self._tensor_args((node.args, node.kwargs))], {}
        self.ops[node.name] = (kind, ins, kw)
        self.graph.add(kind, ins, node.name, name=node.name, **kw)

    def _what(self, node) -> str:
        if node.op == "call_module":
            return type(self.mods[node.target]).__name__
        return _name(node.target)

    def _in(self, node, a, rank=None) -> str:
        """a tensor input: an fx node already in the graph, or a constant made from a python value"""
        import torch.fx as fx
        if isinstance(a, fx.Node):
            if a.name in self.alias:
                return self.alias[a.name]
            if a.name not in self.graph.tensors:
                raise KeyError(f"input {a.name} is not a tensor of the graph")
            return a.name
        return self._const(f"{node.name}.c{len(self.consts)}", a, rank)

    def graph_kinds(self):
        from .dnn_graph import GOP
        return GOP

    def _pv(self, v):
        """a python value an fx arg stands for (traced shape arithmetic), slices and tuples included"""
        import torch.fx as fx
        if isinstance(v, fx.Node):
            if v.name not in self.pyvals:
                raise KeyError(f"{v.name} is not a python value")
            return self.pyvals[v.name]
        if isinstance(v, slice):
            return slice(self._pv(v.start), self._pv(v.stop), self._pv(v.step))
        if isinstance(v, (tuple, list)):
            return type(v)(self._pv(x) for x in v)
        return v

    def _getitem(self, node, meta) -> bool:
        """getitem as the graph's slice (+ a reshape when integer indices drop axes): x[...], or output i
        of x.split / x.chunk (whose tuple is not a graph tensor). False: not a slice this maps."""
        import torch.fx as fx
        src, idx = node.args
        idx = self._pv(idx)
        out = tuple(int(d) for d in meta.shape)
        if isinstance(src, fx.Node) and src.name in self.graph.tensors:      # x[idx]
            shape = tuple(int(d) for d in src.meta["tensor_meta"].shape)
            idx = idx if isinstance(idx, tuple) else (idx,)
            if any(i is None for i in idx) or sum(i is Ellipsis for i in idx) > 1:
                return False
            if Ellipsis in idx:
                e = idx.index(Ellipsis)
                idx = idx[:e] + (slice(None),) * (len(shape) - len(idx) + 1) + idx[e + 1:]
            idx = idx + (slice(None),) * (len(shape) - len(idx))
            starts, sizes = [], []
            for d, i in zip(shape, idx):
                if isinstance(i, int):
                    i = i % d
                    starts.append(i), sizes.append(1)
                elif isinstance(i, slice) and i.step in (None, 1):
                    a, b, _ = i.indices(d)
                    starts.append(a), sizes.append(max(b - a, 0))
                else:
                    return False
            x = src.name
        elif isinstance(src, fx.Node) and isinstance(idx, int) and src.op in ("call_method", "call_function") and \
                _name(src.target) in ("split", "chunk", "split_with_sizes"):    # x.split(...)[i]
            x_node = src.args[0]
            if not isinstance(x_node, fx.Node) or x_node.name not in self.graph.tensors:
                return False
            shape = tuple(int(d) for d in x_node.meta["tensor_meta"].shape)
            kw = {k: self._pv(v) for k, v in src.kwargs.items()}
            sargs = [self._pv(a) if i else a for i, a in enumerate(src.args)]
            dim = kw.get("dim", sargs[2] if len(sargs) > 2 else 0) % len(shape)
            arg = kw.get("split_size_or_sections", kw.get("chunks", sargs[1] if len(sargs) > 1 else None))
            if _name(src.target) == "chunk":
                step = -(-shape[dim] // int(arg))
                sec = [min(step, shape[dim] - k * step) for k in range(int(arg)) if k * step < shape[dim]]
            elif isinstance(arg, int):
                sec = [min(arg, shape[dim] - k) for k in range(0, shape[dim], arg)]
            else:
                sec = [int(v) for v in arg]
            starts = [0] * len(shape)
            starts[dim] = sum(sec[:idx])
            sizes = list(shape)
            sizes[dim] = sec[idx]
            x = x_node.name
        else:
            return False
        if len(starts) > 6:
            return False
        dt = _TORCH_DT.get(str(meta.dtype), "f32")
        if tuple(sizes) == out:
            self.ops[node.name] = ("slice", [x], {"starts": tuple(starts)})
            self.graph.add("slice", [x], node.name, name=node.name, starts=tuple(starts))
        else:   # integer indices: slice keeping the axis, then drop it
            mid = node.name + ".slice"
            self.graph.tensor(mid, tuple(sizes), dt)
            self.ops[mid] = ("slice", [x], {"starts": tuple(starts)})
            self.graph.add("slice", [x], mid, name=mid, starts=tuple(starts))
            self.ops[node.name] = ("reshape", [mid], {})
            self.graph.add("reshape", [mid], node.name, name=node.name)
        return True

    def _aten_slice(self, node, meta) -> bool:
        """aten.slice.Tensor(x, dim, start, end, step=1) / aten.select.int(x, dim, index) as the graph's slice
        (+ a reshape dropping the selected axis)"""
        x = node.args[0]
        if not hasattr(x, "name") or x.name not in self.graph.tensors:
            return False
        shape = [int(d) for d in x.meta["tensor_meta"].shape]
        args = [self._pv(a) for a in node.args[1:]]
        dim = (args[0] if args else 0) % len(shape)
        starts, sizes = [0] * len(shape), list(shape)
        if _name(node.target) == "select":
            starts[dim], sizes[dim] = args[1] % shape[dim], 1
        else:
            a, b, st = slice(args[1] if len(args) > 1 else None, args[2] if len(args) > 2 else None,
                             args[3] if len(args) > 3 else None).indices(shape[dim])
            if st != 1:
                return False
            starts[dim], sizes[dim] = a, max(b - a, 0)
        dt = _TORCH_DT.get(str(meta.dtype), "f32")
        out = tuple(int(d) for d in meta.shape)
        if tuple(sizes) == out:
            self.ops[node.name] = ("slice", [x.name], {"starts": tuple(starts)})
            self.graph.add("slice", [x.name], node.name, name=node.name, starts=tuple(starts))
        else:
            mid = node.name + ".slice"
            self.graph.tensor(mid, tuple(sizes), dt)
            self.ops[mid] = ("slice", [x.name], {"starts": tuple(starts)})
            self.graph.add("slice", [x.name], mid, name=mid, starts=tuple(starts))
            self.ops[node.name] = ("reshape", [mid], {})
            self.graph.add("reshape", [mid], node.name, name=node.name)
        return True

    def _gqa_repeat(self, node):
        """x.repeat_interleave(r, dim=1) whose every reader is an sdpa's K or V: the input's name (the
        graph's sdpa takes K / V with fewer heads than Q, H % Hk == 0, repeated the same way), else None"""
        import torch.fx as fx
        if node.op != "call_method" and not (node.op == "call_function" and _name(node.target) == "repeat_interleave"):
            return None
        if node.op == "call_method" and node.target != "repeat_interleave":
            return None
        args, kw = list(node.args), dict(node.kwargs)
        x, reps, dim = args[0], kw.get("repeats", args[1] if len(args) > 1 else None), kw.get("dim", args[2] if len(args) > 2 else None)
        if not isinstance(x, fx.Node) or x.name not in self.graph.tensors or not isinstance(reps, int) or dim is None:
            return None
        if len(x.meta["tensor_meta"].shape) != 4 or dim % 4 != 1 or not node.users:
            return None
        for u in node.users:
            if _name(u.target) != "scaled_dot_product_attention" or (u.args[0] is node) or \
                    not any(u.args[i] is node for i in (1, 2) if len(u.args) > i):
                return None
        return x.name

    def _map(self, node):
        """(graph-op kind, input names, attrs) for one fx node, or (None, [], {why}) -> wildcard"""
        if node.op == "call_module":
            return self._map_module(node, self.mods[node.target])
        name = _name(node.target) if node.op == "call_function" else node.target
        if node.target is operator.getitem:
            return None, [], {"why": "getitem of a tensor"}
        return self._map_call(node, name, list(node.args), dict(node.kwargs))

    def _eltwise(self, alg, a, b):
        if alg in _KIND:
            kw = {"alpha": a, "beta": b} if alg in ("clip", "relu") else {}
            if alg == "relu" and not a:
                kw = {}
            return _KIND[alg], kw
        return "eltwise", {"alg": "eltwise_" + alg, "alpha": float(a), "beta": float(b)}

    def _map_module(self, node, m):
        from .dnn_ctypes import conv_params, pool_params
        cls, x = type(m).__name__, node.args[0]
        rank = len(node.meta["tensor_meta"].shape)
        if cls in _IDENTITY_MOD:
            return "reshape", [self._in(node, x)], {}
        if cls in _ELTWISE_MOD:
            alg, fa, fb = _ELTWISE_MOD[cls]
            kind, kw = self._eltwise(alg, float(fa(m)), float(fb(m)))
            if cls == "GELU" and m.approximate == "tanh":
                kw = {"alg": "eltwise_gelu_tanh"}
            return kind, [self._in(node, x)], kw
        p = node.name
        if cls == "Conv2d":
            if isinstance(m.padding, str) or m.padding_mode != "zeros":
                return None, [], {"why": f"Conv2d padding {m.padding!r}/{m.padding_mode}"}
            ins = [self._in(node, x), self._const(p + ".weight", m.weight)]
            if m.bias is not None:
                ins.append(self._const(p + ".bias", m.bias))
            pad, dil = _pair(m.padding), _pair(m.dilation)
            return "convolution", ins, {"conv": conv_params(_pair(m.stride), pad, pad, (dil[0] - 1, dil[1] - 1), m.groups)}
        if cls == "ConvTranspose2d":
            return self._deconv(node, x, m)
        if cls == "GroupNorm":
            ins = [self._in(node, x)]
            if m.affine:
                ins += [self._const(p + ".weight", m.weight), self._const(p + ".bias", m.bias)]
            return "group_norm", ins, {"groups": int(m.num_groups), "eps": float(m.eps)}
        if cls == "PReLU":   # slope [C] along dim 1: trailing 1s so it broadcasts after leading 1s
            sl = m.weight.detach().cpu().numpy().reshape((-1,) + (1,) * max(rank - 2, 0))
            return "prelu", [self._in(node, x), self._const(p + ".weight", sl)], {}
        if cls in ("Upsample", "UpsamplingNearest2d", "UpsamplingBilinear2d"):
            mode = {"UpsamplingNearest2d": "nearest", "UpsamplingBilinear2d": "bilinear"}.get(cls, m.mode)
            ac = True if cls == "UpsamplingBilinear2d" else m.align_corners
            return self._resample(node, x, mode, ac)
        if cls == "Linear":
            ins = [self._in(node, x), self._const(p + ".weight", m.weight)]
            if m.bias is not None:
                ins.append(self._const(p + ".bias", m.bias))
            return "matmul", ins, {"weights_transposed": 1}
        if cls in ("BatchNorm2d", "BatchNorm1d"):
            ins = [self._in(node, x), self._const(p + ".mean", m.running_mean), self._const(p + ".var", m.running_var)]
            if m.affine:
                ins += [self._const(p + ".weight", m.weight), self._const(p + ".bias", m.bias)]
            return "batch_norm_inference", ins, {"eps": float(m.eps)}
        if cls == "LayerNorm":
            if len(m.normalized_shape) != 1:
                return None, [], {"why": f"LayerNorm over {tuple(m.normalized_shape)}"}
            ins = [self._in(node, x)]
            if m.elementwise_affine:
                ins += [self._const(p + ".weight", m.weight), self._const(p + ".bias", m.bias)]
            return "layer_norm", ins, {"eps": float(m.eps)}
        if cls in ("MaxPool2d", "AvgPool2d"):
            return self._pool(node, x, cls == "MaxPool2d", m.kernel_size, m.stride, m.padding,
                              getattr(m, "dilation", 1), m.ceil_mode, getattr(m, "count_include_pad", True))
        if cls == "AdaptiveAvgPool2d":
            return self._global_pool(node, x, m.output_size)
        if cls == "Softmax":
            return "softmax", [self._in(node, x)], {"axis": (m.dim if m.dim is not None else -1) % rank}
        return None, [], {"why": cls}

    def _deconv(self, node, x, m):
        from .dnn_ctypes import conv_params
        wt = m.weight.detach().cpu().numpy()              # [Cin, Cout/g, kh, kw] -> aiednn [O, C/g, KH, KW]
        g, cg = m.groups, wt.shape[0] // m.groups
        w = np.concatenate([wt[i * cg:(i + 1) * cg].transpose(1, 0, 2, 3) for i in range(g)], 0)
        pr = tuple(p - op for p, op in zip(_pair(m.padding), _pair(m.output_padding)))
        if min(pr) < 0 or m.padding_mode != "zeros":
            return None, [], {"why": "ConvTranspose2d with output_padding > padding"}
        ins = [self._in(node, x), self._const(node.name + ".weight", w)]
        if m.bias is not None:
            ins.append(self._const(node.name + ".bias", m.bias))
        dil = _pair(m.dilation)
        return "deconvolution", ins, {"conv": conv_params(_pair(m.stride), _pair(m.padding), pr,
                                                          (dil[0] - 1, dil[1] - 1), g)}

    def _resample(self, node, x, mode, align_corners):
        src, dst = x.meta["tensor_meta"].shape[2:], node.meta["tensor_meta"].shape[2:]
        if mode == "nearest" and any(int(o) % int(i) for o, i in zip(dst, src)):
            return None, [], {"why": "'nearest' with a non-integer scale"}   # torch nearest = floor, not half-pixel
        if mode in ("nearest", "nearest-exact"):
            return "resampling", [self._in(node, x)], {"alg": "resampling_nearest"}
        if mode in ("linear", "bilinear") and not align_corners:
            return "resampling", [self._in(node, x)], {"alg": "resampling_linear"}
        return None, [], {"why": f"interpolate {mode} align_corners={align_corners}"}

    def _value(self, a):
        """a python value, or the tensor behind a get_attr node"""
        import torch.fx as fx
        if isinstance(a, fx.Node):
            if a.op != "get_attr":
                raise ValueError(f"{a.name} is not a constant")
            return self._attr(a.target)
        return a

    def _quant(self, node, name, args, kw):
        """PT2E quantize / dequantize (quantized_decomposed.*): per tensor, or per channel for weights"""
        import torch
        if name == "dequantize_per_channel":
            x, sc, zp, axis, qmin, qmax, dt = (list(args) + [None] * 7)[:7]
            sc, zp = self._value(sc), self._value(zp)
            sc = np.asarray(sc.detach().cpu() if isinstance(sc, torch.Tensor) else sc, np.float32).reshape(-1)
            zp = np.asarray(zp.detach().cpu() if isinstance(zp, torch.Tensor) else zp, np.int64).reshape(-1)
            return "dequantize", [self._in(node, x)], {
                "scales": sc, "zero_points": np.broadcast_to(zp, sc.shape).astype(np.int32), "axis": int(axis)}
        x, sc, zp, qmin, qmax, dt = (list(args) + [None] * 6)[:6]
        sc, zp = float(self._value(sc)), int(self._value(zp))
        info = torch.iinfo(dt) if dt is not None else None
        if info is None or (qmin, qmax) != (info.min, info.max):
            return None, [], {"why": f"{name} with a narrowed range [{qmin}, {qmax}]"}
        return ("quantize" if name == "quantize_per_tensor" else "dequantize"), [self._in(node, x)], {
            "scale": sc, "zero_point": zp}

    def _conv_call(self, node, name, args, kw):
        """aten.conv2d / convolution as a function: (x, w, b, stride, padding, dilation, [transposed,
        output_padding,] groups) with tensors for w and b (dequantize outputs under PT2E)"""
        from .dnn_ctypes import conv_params
        a = list(args) + [None] * 9
        x, w, b = a[0], a[1], a[2]
        if name == "convolution":
            if a[6]:
                return None, [], {"why": "transposed aten.convolution"}
            st, pad, dil, groups = a[3], a[4], a[5], a[8]
        else:
            st, pad, dil, groups = a[3] or 1, a[4] or 0, a[5] or 1, a[6] or 1
        if isinstance(pad, str) or len(node.meta["tensor_meta"].shape) != 4:
            return None, [], {"why": f"conv padding {pad!r} / not 2-D"}
        pad, dil = _pair(pad), _pair(dil)
        ins = [self._in(node, x), self._in(node, w)] + ([self._in(node, b)] if b is not None else [])
        return "convolution", ins, {"conv": conv_params(_pair(st), pad, pad, (dil[0] - 1, dil[1] - 1), int(groups))}

    def _reduce(self, node, name, args, kw):
        x = args[0]
        r = len(x.meta["tensor_meta"].shape)
        dims = kw.get("dim", args[1] if len(args) > 1 else None)
        dims = list(range(r)) if dims is None else [dims] if isinstance(dims, int) else list(dims)
        mask = sum(1 << (d % r) for d in dims)
        alg = {"mean": "reduction_mean", "sum": "reduction_sum", "amax": "reduction_max", "amin": "reduction_min"}[name]
        return "reduction", [self._in(node, x)], {"alg": alg, "reduce_mask": mask}

    def _pool(self, node, x, is_max, k, s, pad, dil, ceil, include_pad):
        from .dnn_ctypes import pool_params
        k, s, pad, dil = _pair(k), _pair(s or k), _pair(pad), _pair(dil)
        pad_r = pad
        if ceil:   # the extra windows as right padding: exact for max (padding never wins), not for avg
            if not is_max:
                return None, [], {"why": "avg pooling with ceil_mode"}
            src, dst = x.meta["tensor_meta"].shape[2:], node.meta["tensor_meta"].shape[2:]
            pad_r = tuple(max(pad[i], (int(dst[i]) - 1) * s[i] + (k[i] - 1) * dil[i] + 1 - int(src[i]) - pad[i])
                          for i in range(2))
        kw = {"pool": pool_params(k, s, pad, pad_r, (dil[0] - 1, dil[1] - 1))}
        if not is_max and not include_pad:
            kw["alg"] = "pooling_avg_exclude_padding"
        return ("max_pool" if is_max else "avg_pool"), [self._in(node, x)], kw

    def _global_pool(self, node, x, size):
        from .dnn_ctypes import pool_params
        if _pair(size) != (1, 1):
            return None, [], {"why": f"adaptive pooling to {size}"}
        h, w = (int(d) for d in x.meta["tensor_meta"].shape[2:])
        return "avg_pool", [self._in(node, x)], {"pool": pool_params((h, w), (1, 1))}

    def _map_call(self, node, name, args, kw):
        rank = len(node.meta["tensor_meta"].shape)
        if name.endswith("_") and not name.endswith("__") and name not in _ELTWISE_FN:
            # in-place aten op (add_, hardtanh_): its functional form, when nothing else reads the
            # tensor it mutates (else a later reader would see the mutation)
            src = args[0] if args else None
            if src is None or not hasattr(src, "users") or len(src.users) != 1:
                return None, [], {"why": f"in-place {name} on a tensor with other readers"}
            name = name[:-1]
        if name in _ALIAS_FN or name in ("unflatten", "_unsafe_view"):
            return "reshape", [self._in(node, args[0])], {}
        if name in ("upsample_nearest2d", "upsample_bilinear2d"):   # aten (PT2E) forms of interpolate
            ac = args[2] if name == "upsample_bilinear2d" and len(args) > 2 else None
            return self._resample(node, args[0], "nearest" if "nearest" in name else "bilinear", ac)
        if name in _BIN:
            a, b = args[0], kw.get("other", args[1] if len(args) > 1 else None)
            if kw.get("alpha", 1) != 1:
                return None, [], {"why": f"{name} with alpha"}
            kind = _BIN[name]
            ins = [self._in(node, a, rank), self._in(node, b, rank)]
            if kind.startswith("binary_"):
                return "binary", ins, {"alg": kind}
            return kind, ins, {}
        if name in _ELTWISE_FN:
            kind, kwa = self._eltwise(*_ELTWISE_FN[name])
            return kind, [self._in(node, args[0])], kwa
        if name == "gelu":
            return "gelu", [self._in(node, args[0])], (
                {"alg": "eltwise_gelu_tanh"} if kw.get("approximate", "none") == "tanh" else {})
        if name in ("hardtanh", "clamp", "clip"):
            lo = kw.get("min", kw.get("min_val", args[1] if len(args) > 1 else -1.0))
            hi = kw.get("max", kw.get("max_val", args[2] if len(args) > 2 else 1.0))
            if lo is None or hi is None:
                return None, [], {"why": f"{name} with one bound"}
            return "clamp", [self._in(node, args[0])], {"alpha": float(lo), "beta": float(hi)}
        if name in ("permute", "transpose"):
            r = len(args[0].meta["tensor_meta"].shape)
            if name == "permute":
                perm = args[1:] if not isinstance(args[1], (list, tuple)) else list(args[1])
            else:
                perm = list(range(r))
                i, j = args[1] % r, args[2] % r
                perm[i], perm[j] = perm[j], perm[i]
            return "transpose", [self._in(node, args[0])], {"perm": [int(p) % r for p in perm]}
        if name in ("cat", "concat"):
            dim = kw.get("dim", args[1] if len(args) > 1 else 0)
            return "concat", [self._in(node, t) for t in args[0]], {"axis": dim % rank}
        if name in ("softmax", "_softmax", "log_softmax", "_log_softmax"):
            dim = kw.get("dim", args[1] if len(args) > 1 else -1)
            kw2 = {"axis": dim % rank}
            if "log" in name:
                kw2["alg"] = "softmax_log"
            return "softmax", [self._in(node, args[0])], kw2
        kinds = self.graph_kinds()
        if name == "where" and "select" in kinds and len(args) == 3:   # where(cond, a, b): the graph's select
            return "select", [self._in(node, a, rank) for a in args], {}
        if name in ("_to_copy", "to", "type_as") and "type_cast" in kinds:
            src = args[0]
            if hasattr(src, "meta") and str(src.meta["tensor_meta"].dtype) == str(node.meta["tensor_meta"].dtype):
                return "reshape", [self._in(node, src)], {}   # a copy, no cast
            return "type_cast", [self._in(node, src)], {}
        if name == "rms_norm" and "rms_norm" in kinds:   # aten.rms_norm(x, normalized_shape, weight, eps)
            x = args[0]
            w = kw.get("weight", args[2] if len(args) > 2 else None)
            eps = kw.get("eps", args[3] if len(args) > 3 else None)
            ins = [self._in(node, x)] + ([self._in(node, w)] if w is not None else [])
            return "rms_norm", ins, {"eps": float(eps if eps is not None else 1e-6)}
        if name in ("adaptive_avg_pool2d",):
            return self._global_pool(node, args[0], kw.get("output_size", args[1] if len(args) > 1 else None))
        if name in ("max_pool2d", "avg_pool2d"):
            def arg(i, key, dflt):
                return kw.get(key, args[i] if len(args) > i else dflt)
            k = arg(1, "kernel_size", None)
            if name == "max_pool2d":
                return self._pool(node, args[0], True, k, arg(2, "stride", None), arg(3, "padding", 0),
                                  arg(4, "dilation", 1), arg(5, "ceil_mode", False), True)
            if arg(6, "divisor_override", None):
                return None, [], {"why": "divisor_override"}
            return self._pool(node, args[0], False, k, arg(2, "stride", None), arg(3, "padding", 0), 1,
                              arg(4, "ceil_mode", False), arg(5, "count_include_pad", True))
        if name in ("quantize_per_tensor", "dequantize_per_tensor", "dequantize_per_channel"):
            return self._quant(node, name, args, kw)
        if name in ("conv2d", "convolution") and len(args) >= 2:
            return self._conv_call(node, name, args, kw)
        if name in ("mean", "sum", "amax", "amin") and isinstance(args[0], self.torch.fx.Node):
            if kw.get("dtype") is not None:
                return None, [], {"why": f"{name} with dtype"}
            return self._reduce(node, name, args, kw)
        if name == "interpolate":
            return self._resample(node, args[0], kw.get("mode", "nearest"), kw.get("align_corners"))
        if name == "layer_norm":
            x = args[0]
            shape = kw.get("normalized_shape", args[1] if len(args) > 1 else None)
            w = kw.get("weight", args[2] if len(args) > 2 else None)
            b = kw.get("bias", args[3] if len(args) > 3 else None)
            eps = kw.get("eps", args[4] if len(args) > 4 else 1e-5)
            if len(shape) != 1 or int(shape[0]) != int(x.meta["tensor_meta"].shape[-1]) or (w is None) != (b is None):
                return None, [], {"why": f"layer_norm over {tuple(shape)}"}
            ins = [self._in(node, x)] + ([self._in(node, w), self._in(node, b)] if w is not None else [])
            return "layer_norm", ins, {"eps": float(eps)}
        if name in ("matmul", "bmm"):
            return "matmul", [self._in(node, args[0], rank), self._in(node, args[1], rank)], {}
        if name == "linear":
            w = args[1] if len(args) > 1 else kw["weight"]
            b = kw.get("bias", args[2] if len(args) > 2 else None)
            ins = [self._in(node, args[0]), self._in(node, w)] + ([self._in(node, b)] if b is not None else [])
            return "matmul", ins, {"weights_transposed": 1}
        if name == "scaled_dot_product_attention":
            q, mask = args[0], kw.get("attn_mask", args[3] if len(args) > 3 else None)
            if kw.get("dropout_p", args[4] if len(args) > 4 else 0.0):
                return None, [], {"why": "sdpa with dropout"}
            if mask is not None and (not hasattr(mask, "meta") or str(mask.meta["tensor_meta"].dtype) != "torch.float32"):
                return None, [], {"why": "sdpa with a non-float (boolean) mask"}
            d = int(q.meta["tensor_meta"].shape[-1])
            ins = [self._in(node, t) for t in args[:3]] + ([self._in(node, mask)] if mask is not None else [])
            return "sdpa", ins, {   # an additive float mask: the graph sdpa's 4th input
                # aten order (q, k, v, attn_mask, dropout_p, is_causal, scale): PT2E passes them positionally
                "sdpa_scale": float(kw.get("scale", args[6] if len(args) > 6 else None) or 1.0 / math.sqrt(d)),
                "causal": int(bool(kw.get("is_causal", args[5] if len(args) > 5 else False)))}
        if name == "pow" and isinstance(kw.get("exponent", args[1] if len(args) > 1 else None), (int, float)):
            e = float(kw.get("exponent", args[1] if len(args) > 1 else 1))
            return "eltwise", [self._in(node, args[0])], {"alg": "eltwise_pow", "alpha": 1.0, "beta": e}
        if name == "expand" and tuple(args[0].meta["tensor_meta"].shape) == tuple(node.meta["tensor_meta"].shape):
            return "reshape", [self._in(node, args[0])], {}   # an expand that broadcasts nothing (batch 1)
        return None, [], {"why": name}

    # ── reporting ────────────────────────────────────────────────────────
    def report(self) -> str:
        sup = [p for p in self.parts if p.supported]
        lines = [f"{len(self.parts)} partitions: {len(sup)} supported, {len(self.parts) - len(sup)} unsupported; "
                 f"{sum(len(p.ops) for p in sup)} / {len(self.ops)} fx nodes inside supported partitions"]
        if self.wildcards:
            lines.append("  no mapping: " + ", ".join(f"{k} ({v})" for k, v in self.wildcards.items()))
        lines += [f"  {'+' if p.supported else '-'} {p.info}" for p in self.parts]
        return "\n".join(lines)
