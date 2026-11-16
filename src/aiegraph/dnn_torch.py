###############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
"""PyTorch ``nn.Module`` -> aiednn primitives: lower once, run on x86 or the board."""

from __future__ import annotations

import math
import operator
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .errors import AiegraphError

__all__ = ["AGraph", "lower", "run_ctypes", "emit_app", "fake_quant_reference", "model_test", "ModelReport"]

DT_NP = {"f32": np.float32, "s8": np.int8, "s32": np.int32, "u8": np.uint8}
DT_C = {"f32": "float", "s8": "int8_t", "s32": "int32_t", "u8": "uint8_t"}
DT_ENUM = {"s8": "aiednn_s8", "u8": "aiednn_u8", "s32": "aiednn_s32", "f32": "aiednn_f32"}


# ── the lowered graph ─────────────────────────────────────────────────────

@dataclass
class Tensor:
    id: int
    shape: tuple
    dt: str = "f32"
    kind: str = "act"          # act | const | input
    data: np.ndarray = None    # const / input
    base: int = None           # alias of another tensor (same bytes, other shape)


@dataclass
class Op:
    name: str                  # the fx node it came from
    fn: str                    # aiednn_*_create
    create: list               # ("md", tid) | ("null",) | ("int"|"i64"|"uint"|"float", v) | ("alg", name)
                               # | ("conv", dict) | ("pool", dict) | ("perm", list) | ("mds", [tid]) | ("floats", list) | ("i64s", list)
    exec: list                 # (AIEDNN_ARG name | ("multi", i), tid)
    out: int


@dataclass
class AGraph:
    name: str = "model"
    mode: str = "f32"
    tensors: list = field(default_factory=list)
    ops: list = field(default_factory=list)
    input: int = None
    output: int = None
    nodes: dict = field(default_factory=dict)  # fx node name -> tensor id (per-layer debugging)
    plan: dict = None      # tensor id -> arena byte offset (--aiednn-plan-memory; None: _plan_memory)
    arena: int = None      # arena bytes of that plan
    top1_range: tuple = None   # (lo, hi, group): top-1 = (argmax over [lo, hi) - lo) // group per image

    def new(self, shape, dt="f32", kind="act", data=None, base=None) -> int:
        self.tensors.append(Tensor(len(self.tensors), tuple(int(d) for d in shape), dt, kind, data, base))
        return len(self.tensors) - 1

    def const(self, arr, dt="f32") -> int:
        a = np.ascontiguousarray(np.asarray(arr), DT_NP[dt])
        return self.new(a.shape if a.ndim else (1,), dt, "const", a.reshape(a.shape if a.ndim else (1,)))

    def alias(self, tid, shape) -> int:
        t = self.tensors[tid]
        if int(np.prod(shape)) != int(np.prod(t.shape)):
            raise AiegraphError("dnn_torch", f"view {t.shape} -> {tuple(shape)} changes the element count")
        if t.kind == "const":
            return self.const(t.data.reshape(shape), t.dt)
        return self.new(shape, t.dt, t.kind, None, self.root(tid))

    def root(self, tid) -> int:
        while self.tensors[tid].base is not None:
            tid = self.tensors[tid].base
        return tid

    def op(self, name, fn, create, exec_, out) -> int:
        self.ops.append(Op(name, fn, create, exec_, out))
        return out

    def summary(self) -> str:
        kinds = {}
        for o in self.ops:
            k = o.fn.replace("aiednn_", "").replace("_forward", "").replace("_create", "").replace("_inference", "")
            kinds[k] = kinds.get(k, 0) + 1
        nconst = sum(t.data.nbytes for t in self.tensors if t.kind == "const")
        return (f"{self.name} [{self.mode}]: {len(self.ops)} primitives "
                f"({', '.join(f'{k} {n}' for k, n in sorted(kinds.items(), key=lambda kv: -kv[1]))}); "
                f"constants {nconst / 1e6:.1f} MB")


# ── lowering ──────────────────────────────────────────────────────────────

_ELTWISE_MOD = {  # module class name -> (alg, alpha(module), beta(module))
    "ReLU": ("relu", lambda m: 0.0, lambda m: 0.0),
    "ReLU6": ("clip", lambda m: 0.0, lambda m: 6.0),
    "Hardtanh": ("clip", lambda m: m.min_val, lambda m: m.max_val),
    "LeakyReLU": ("relu", lambda m: m.negative_slope, lambda m: 0.0),
    "ELU": ("elu", lambda m: m.alpha, lambda m: 0.0),
    "GELU": ("gelu", lambda m: 0.0, lambda m: 0.0),
    "SiLU": ("swish", lambda m: 1.0, lambda m: 0.0),
    "Sigmoid": ("logistic", lambda m: 0.0, lambda m: 0.0),
    "Tanh": ("tanh", lambda m: 0.0, lambda m: 0.0),
    "Hardswish": ("hardswish", lambda m: 1 / 6, lambda m: 0.5),
    "Hardsigmoid": ("hardsigmoid", lambda m: 1 / 6, lambda m: 0.5),
    "Mish": ("mish", lambda m: 0.0, lambda m: 0.0),
    "Softplus": ("soft_relu", lambda m: m.beta, lambda m: 0.0),
}
_IDENTITY_MOD = {"Identity", "Dropout", "Dropout2d", "StochasticDepth", "Flatten", "Unflatten"}


class _Lowerer:
    def __init__(self, gm, mode: str, qscale: dict):
        self.gm, self.mode, self.qscale = gm, mode, qscale
        self.g = AGraph(mode=mode)
        self.env = {}
        self.mods = dict(gm.named_modules())
        self.fused = {}                 # node absorbed into a GEMM epilogue -> its output tensor
        self.fuse = True

    # helpers ---------------------------------------------------------------
    def shape(self, node) -> tuple:
        tm = node.meta.get("tensor_meta")
        if tm is None or not hasattr(tm, "shape"):
            raise AiegraphError("dnn_torch", f"{node.name}: no tensor shape (multiple outputs?)")
        return tuple(int(d) for d in tm.shape)

    def val(self, a):
        import torch.fx as fx
        if isinstance(a, fx.Node):
            return self.env[a]
        if isinstance(a, (list, tuple)):
            return type(a)(self.val(x) for x in a)
        if isinstance(a, slice):   # x[..., :h] with h a traced shape value
            return slice(self.val(a.start), self.val(a.stop), self.val(a.step))
        return a

    def tid(self, v, rank=None) -> int:
        """an activation id, or a constant from a python scalar / tensor"""
        if isinstance(v, _Tid):
            return int(v)
        import torch
        arr = v.detach().numpy() if isinstance(v, torch.Tensor) else np.asarray(v, np.float32)
        if rank is not None and arr.ndim < rank:
            arr = arr.reshape((1,) * (rank - arr.ndim) + arr.shape)
        return self.g.const(arr.astype(np.float32))

    def rank_to(self, t, rank) -> int:
        s = self.g.tensors[t].shape
        return t if len(s) == rank else self.g.alias(t, (1,) * (rank - len(s)) + s)

    def out(self, node, dt="f32") -> _Tid:
        return _Tid(self.g.new(self.shape(node), dt))

    # primitive emitters ------------------------------------------------------
    def eltwise(self, name, x, alg, alpha, beta, shape=None):
        y = self.g.new(shape or self.g.tensors[x].shape)
        return self.g.op(name, "aiednn_eltwise_forward_create",
                         [("alg", "eltwise_" + alg), ("float", alpha), ("float", beta), ("md", x), ("md", y)],
                         [("SRC", x), ("DST", y)], y)

    def binary(self, name, alg, a, b, shape):
        r = len(shape)
        a, b = self.rank_to(a, r), self.rank_to(b, r)
        y = self.g.new(shape)
        return self.g.op(name, "aiednn_binary_create",
                         [("alg", "binary_" + alg), ("md", a), ("md", b), ("null",), ("md", y)],
                         [("SRC", a), ("SRC_1", b), ("DST", y)], y)

    def reorder(self, name, x, dt, perm=None, scale=1.0, zp=0, shape=None):
        s = self.g.tensors[x].shape
        y = self.g.new(shape or (tuple(s[p] for p in perm) if perm else s), dt)
        return self.g.op(name, "aiednn_reorder_create",
                         [("md", x), ("md", y), ("perm", list(perm)) if perm else ("null",), ("float", scale),
                          ("int", zp)], [("SRC", x), ("DST", y)], y)

    # dispatch ----------------------------------------------------------------
    def run(self, example_input):
        from torch.fx.passes.shape_prop import ShapeProp
        ShapeProp(self.gm).propagate(example_input)
        for node in self.gm.graph.nodes:
            if node in self.fused:
                self.env[node] = self.fused[node]
                self.g.nodes[node.name] = int(self.fused[node])
                continue
            if node.op == "placeholder":
                t = self.g.new(self.shape(node), "f32", "input")
                self.g.input = t
                self.env[node] = _Tid(t)
            elif node.op == "get_attr":
                attr = self.gm
                for part in node.target.split("."):
                    attr = getattr(attr, part)
                self.env[node] = attr
            elif node.op == "output":
                res = self.val(node.args[0])
                res = res[0] if isinstance(res, (list, tuple)) else res
                self.g.output = int(res)
            elif node.op == "call_module":
                self.env[node] = self.module(node, self.mods[node.target])
            elif node.op == "call_function":
                self.env[node] = self.function(node, node.target)
            elif node.op == "call_method":
                self.env[node] = self.method(node, node.target)
            if isinstance(self.env.get(node), _Tid):
                self.g.nodes[node.name] = int(self.env[node])
        return self.g

    # modules -----------------------------------------------------------------
    def module(self, node, m):
        cls = type(m).__name__
        x = self.val(node.args[0])
        if cls in _IDENTITY_MOD:
            return _Tid(self.g.alias(int(x), self.shape(node)))
        if cls in _ELTWISE_MOD:
            alg, fa, fb = _ELTWISE_MOD[cls]
            if cls == "GELU":
                alg = "gelu_tanh" if m.approximate == "tanh" else "gelu_erf"
            return _Tid(self.eltwise(node.name, int(x), alg, float(fa(m)), float(fb(m))))
        handler = getattr(self, f"_mod_{cls}", None)
        if handler is None:
            raise AiegraphError("dnn_torch", f"{node.name}: module {cls} is not supported")
        return _Tid(handler(node, m, int(x)))

    def _mod_Conv2d(self, node, m, x):
        if m.padding_mode != "zeros" or isinstance(m.padding, str):
            raise AiegraphError("dnn_torch", f"{node.name}: padding {m.padding!r}/{m.padding_mode} not supported")
        w = m.weight.detach().numpy()
        b = None if m.bias is None else m.bias.detach().numpy()
        cp = dict(strides=m.stride, pl=m.padding, pr=m.padding, dil=tuple(d - 1 for d in m.dilation), groups=m.groups)
        return self.gemm_like(node, "aiednn_convolution_forward_create", x, w, b, cp, chan_axis=1)

    def _mod_ConvTranspose2d(self, node, m, x):
        wt = m.weight.detach().numpy()              # [Cin, Cout/g, kh, kw]
        g = m.groups
        Cin, Og = wt.shape[0], wt.shape[1]
        Cg = Cin // g
        w = np.concatenate([wt[i * Cg:(i + 1) * Cg].transpose(1, 0, 2, 3) for i in range(g)], 0)  # [O, Cg, kh, kw]
        pr = tuple(p - op for p, op in zip(m.padding, m.output_padding))
        if min(pr) < 0:
            raise AiegraphError("dnn_torch", f"{node.name}: output_padding > padding is not supported")
        cp = dict(strides=m.stride, pl=m.padding, pr=pr, dil=tuple(d - 1 for d in m.dilation), groups=g)
        b = None if m.bias is None else m.bias.detach().numpy()
        return self.gemm_like(node, "aiednn_deconvolution_forward_create", x, w, b, cp, chan_axis=1)

    def _mod_Linear(self, node, m, x):
        return self.linear(node, x, m.weight.detach().numpy(), None if m.bias is None else m.bias.detach().numpy())

    def linear(self, node, x, w, b):
        r = len(self.g.tensors[x].shape)
        if r < 2:
            x, r = self.g.alias(x, (1,) + self.g.tensors[x].shape), 2
        wr = w.reshape((1,) * (r - 2) + w.shape)
        return self.gemm_like(node, "aiednn_matmul_forward_create", x, wr, b, None, chan_axis=r - 1)

    # epilogue fusion ----------------------------------------------------------
    def post_op_of(self, u, prev, shape):
        """the post-op a single user *u* of *prev* folds into, or None"""
        name = None
        if u.op == "call_module":
            m = self.mods[u.target]
            cls = type(m).__name__
            if cls in _ELTWISE_MOD:
                alg, fa, fb = _ELTWISE_MOD[cls]
                if cls == "GELU":
                    alg = "gelu_tanh" if m.approximate == "tanh" else "gelu_erf"
                return ("eltwise", alg, float(fa(m)), float(fb(m)))
            return None
        if u.op in ("call_function", "call_method"):
            name = u.target if u.op == "call_method" else getattr(u.target, "__name__", "")
        if name in _ELTWISE_FN:
            alg, a, b = _ELTWISE_FN[name]
            return ("eltwise", alg, float(a), float(b))
        if name in ("add", "iadd") and len(u.args) == 2 and not u.kwargs:
            other = u.args[1] if u.args[0] is prev else u.args[0]
            import torch.fx as fx
            if not isinstance(other, fx.Node):
                return ("eltwise", "linear", 1.0, float(other)) if isinstance(other, (int, float)) else None
            if other not in self.env or not isinstance(self.env[other], _Tid):
                return None                              # computed after this GEMM: cannot be an input of it
            o = int(self.env[other])
            o = self.rank_to(o, len(shape))
            if not md_bcast(self.g.tensors[o].shape, shape) or self.shape(u) != tuple(shape):
                return None
            return ("binary", "add", o)
        return None

    def absorb(self, node, shape):
        """follow single-user eltwise / residual-add successors of *node* (<= 4)"""
        post, last = [], node
        while self.fuse and len(post) < 4 and len(last.users) == 1:
            u = next(iter(last.users))
            po = self.post_op_of(u, last, shape)
            if po is None:
                break
            post.append((po, u))
            last = u
        return post

    def gemm_like(self, node, fn, x, w, b, cp, chan_axis):
        """conv / deconv / linear in the graph's mode"""
        shape = self.shape(node)
        r = len(shape)
        bshape = [1] * r
        bshape[chan_axis] = shape[chan_axis]
        tail = [("int", 1)] if fn == "aiednn_matmul_forward_create" else [("conv", cp)]
        post = self.absorb(node, shape)
        bias = None
        if b is not None:
            bias = self.g.const((b.reshape(bshape) if fn.startswith("aiednn_matmul") else b).astype(np.float32))
        if self.mode == "f32":
            src, wt, scales = x, self.g.const(w, "f32"), None
        else:
            src, wt, scales = self.quantize_gemm_inputs(node, fn, x, w)
        y = self.g.new(shape)
        attr = dict(scales=scales, post=[po for po, _ in post])
        ex = [("SRC", src), ("WEIGHTS", wt), ("DST", y)] + ([("BIAS", bias)] if bias is not None else [])
        ex += [(("post", k), po[2]) for k, (po, _) in enumerate(post) if po[0] == "binary"]
        has_attr = scales is not None or post
        self.g.op(node.name, fn.replace("_create", "_create_attr") if has_attr else fn,
                  [("md", src), ("md", wt), ("md", bias) if bias is not None else ("null",), ("md", y)] + tail
                  + ([("attr", attr)] if has_attr else []), ex, y)
        for _, u in post:
            self.fused[u] = _Tid(y)
        return y

    def quantize_gemm_inputs(self, node, fn, x, w):
        """int8 mode: quantize src (per tensor), weights (per output channel)"""
        sx = self.qscale[node.target]
        wflat = w.reshape(w.shape[0], -1) if fn != "aiednn_matmul_forward_create" else w.reshape(-1, w.shape[-1])
        sw = np.abs(wflat).max(axis=1) / 127.0
        sw[sw == 0] = 1.0
        wshape = (w.shape[0],) + (1,) * (w.ndim - 1) if fn != "aiednn_matmul_forward_create" else \
            (1,) * (w.ndim - 2) + (w.shape[-2], 1)
        wq = np.clip(np.rint(w / sw.reshape(wshape)), -127, 127).astype(np.int8)
        xq = self.reorder(node.name + "_q", x, "s8", scale=sx)
        return xq, self.g.const(wq, "s8"), (sx * sw).astype(np.float32)

    def _mod_BatchNorm2d(self, node, m, x):
        C = m.num_features
        mean, var = m.running_mean.detach().numpy(), m.running_var.detach().numpy()
        y = self.g.new(self.shape(node))
        ex = [("SRC", x), ("DST", y), ("MEAN", self.g.const(mean)), ("VARIANCE", self.g.const(var))]
        flags = 0
        if m.affine:
            flags = 3
            ex += [("SCALE", self.g.const(m.weight.detach().numpy())), ("SHIFT", self.g.const(m.bias.detach().numpy()))]
        assert C == self.g.tensors[x].shape[1]
        return self.g.op(node.name, "aiednn_batch_normalization_forward_inference_create",
                         [("md", x), ("md", y), ("float", m.eps), ("uint", flags)], ex, y)

    def _norm(self, node, fn, x, eps, weight, bias, extra=None, flags=0):
        y = self.g.new(self.shape(node))
        ex = [("SRC", x), ("DST", y)]
        if weight is not None:
            flags |= 1
            ex.append(("SCALE", self.g.const(weight.detach().numpy())))
        if bias is not None:
            flags |= 2
            ex.append(("SHIFT", self.g.const(bias.detach().numpy())))
        create = ([extra] if extra else []) + [("md", x), ("md", y), ("float", eps), ("uint", flags)]
        return self.g.op(node.name, fn, create, ex, y)

    def _mod_LayerNorm(self, node, m, x):
        if len(m.normalized_shape) != 1:
            raise AiegraphError("dnn_torch", f"{node.name}: LayerNorm over {m.normalized_shape} (last dim only)")
        return self._norm(node, "aiednn_layer_normalization_forward_inference_create", x, m.eps,
                          m.weight if m.elementwise_affine else None,
                          m.bias if m.elementwise_affine and m.bias is not None else None)

    def _mod_GroupNorm(self, node, m, x):
        return self._norm(node, "aiednn_group_normalization_forward_inference_create", x, m.eps,
                          m.weight if m.affine else None, m.bias if m.affine else None,
                          extra=("i64", m.num_groups))

    def _mod_MaxPool2d(self, node, m, x):
        return self.pool(node, x, "max", m.kernel_size, m.stride, m.padding, m.dilation, m.ceil_mode)

    def _mod_AvgPool2d(self, node, m, x):
        if m.divisor_override:
            raise AiegraphError("dnn_torch", f"{node.name}: divisor_override not supported")
        alg = "avg_include_padding" if m.count_include_pad else "avg_exclude_padding"
        return self.pool(node, x, alg, m.kernel_size, m.stride, m.padding, 1, m.ceil_mode)

    def _mod_AdaptiveAvgPool2d(self, node, m, x):
        return self.adaptive_avg(node, x)

    def _mod_Upsample(self, node, m, x):
        return self.resample(node, x, m.mode, m.align_corners)

    def _mod_Softmax(self, node, m, x):
        return self.softmax(node, x, m.dim, "accurate")

    def _mod_LogSoftmax(self, node, m, x):
        return self.softmax(node, x, m.dim, "log")

    def _mod_PReLU(self, node, m, x):
        shape = self.shape(node)
        w = m.weight.detach().numpy()
        ws = [1] * len(shape)
        if w.size > 1:
            ws[1] = w.size
        wt = self.g.const(w.reshape(ws))
        y = self.g.new(shape)
        return self.g.op(node.name, "aiednn_prelu_forward_create", [("md", x), ("md", wt), ("md", y)],
                         [("SRC", x), ("WEIGHTS", wt), ("DST", y)], y)

    def _mod_LocalResponseNorm(self, node, m, x):
        y = self.g.new(self.shape(node))
        return self.g.op(node.name, "aiednn_lrn_forward_create",
                         [("alg", "lrn_across_channels"), ("i64", m.size), ("float", m.alpha), ("float", m.beta),
                          ("float", m.k), ("md", x), ("md", y)], [("SRC", x), ("DST", y)], y)

    def _mod_ChannelShuffle(self, node, m, x):
        y = self.g.new(self.shape(node))
        return self.g.op(node.name, "aiednn_shuffle_forward_create", [("int", 1), ("i64", m.groups), ("md", x), ("md", y)],
                         [("SRC", x), ("DST", y)], y)

    def _mod_Permute(self, node, m, x):                # torchvision.ops.misc.Permute
        return self.reorder(node.name, x, "f32", perm=list(m.dims))

    # spatial helpers ---------------------------------------------------------
    def pool(self, node, x, alg, k, s, p, d, ceil):
        pair = lambda v: (v, v) if isinstance(v, int) else tuple(v)  # noqa: E731
        k, s, p, d = pair(k), pair(s if s else k), pair(p), pair(d)
        shape, src = self.shape(node), self.g.tensors[x].shape
        pr = list(p)
        for i in range(2):
            ext = (k[i] - 1) * d[i] + 1
            floor = (src[2 + i] + 2 * p[i] - ext) // s[i] + 1
            if shape[2 + i] != floor:                  # ceil_mode: widen the right padding
                if alg != "max":
                    raise AiegraphError("dnn_torch", f"{node.name}: avg pooling with ceil_mode")
                pr[i] = (shape[2 + i] - 1) * s[i] + ext - src[2 + i] - p[i]
        pp = dict(kernel=k, strides=s, pl=p, pr=tuple(pr), dil=tuple(v - 1 for v in d))
        y = self.g.new(shape)
        return self.g.op(node.name, "aiednn_pooling_forward_create",
                         [("alg", "pooling_" + alg), ("md", x), ("md", y), ("pool", pp)], [("SRC", x), ("DST", y)], y)

    def adaptive_avg(self, node, x):
        shape, src = self.shape(node), self.g.tensors[x].shape
        if shape[2:] == src[2:]:
            return self.g.alias(x, shape)
        if shape[2:] == (1, 1):
            y = self.g.new(shape)
            return self.g.op(node.name, "aiednn_reduction_create",
                             [("alg", "reduction_mean"), ("float", 0.0), ("float", 0.0), ("md", x), ("md", y)],
                             [("SRC", x), ("DST", y)], y)
        if src[2] % shape[2] or src[3] % shape[3]:
            raise AiegraphError("dnn_torch", f"{node.name}: adaptive pooling {src[2:]} -> {shape[2:]}")
        k = (src[2] // shape[2], src[3] // shape[3])
        return self.pool(node, x, "avg_include_padding", k, k, 0, 1, False)

    def resample(self, node, x, mode, align_corners):
        shape, src = self.shape(node), self.g.tensors[x].shape
        if mode == "nearest":
            if any(o % i for o, i in zip(shape[2:], src[2:])):   # torch 'nearest' == half-pixel only for x k
                raise AiegraphError("dnn_torch", f"{node.name}: 'nearest' with a non-integer scale")
            alg = "nearest"
        elif mode in ("nearest-exact",):
            alg = "nearest"
        elif mode in ("linear", "bilinear", "trilinear") and not align_corners:
            alg = "linear"
        else:
            raise AiegraphError("dnn_torch", f"{node.name}: interpolate mode {mode} align_corners={align_corners}")
        y = self.g.new(shape)
        return self.g.op(node.name, "aiednn_resampling_forward_create", [("alg", "resampling_" + alg), ("md", x), ("md", y)],
                         [("SRC", x), ("DST", y)], y)

    def softmax(self, node, x, dim, alg):
        r = len(self.g.tensors[x].shape)
        y = self.g.new(self.shape(node))
        return self.g.op(node.name, "aiednn_softmax_forward_create",
                         [("alg", "softmax_" + alg), ("int", dim % r), ("md", x), ("md", y)], [("SRC", x), ("DST", y)], y)

    # functions / methods -----------------------------------------------------
    def function(self, node, fn):
        return _lower_call(self, node, getattr(fn, "__name__", str(fn)), [self.val(a) for a in node.args],
                           {k: self.val(v) for k, v in node.kwargs.items()})

    def method(self, node, name):
        return _lower_call(self, node, name, [self.val(a) for a in node.args],
                           {k: self.val(v) for k, v in node.kwargs.items()})


class _Tid(int):
    """a tensor id (distinguishes graph tensors from python ints in node args)"""


def md_bcast(src: tuple, dst: tuple) -> bool:
    return len(src) == len(dst) and all(s == d or s == 1 for s, d in zip(src, dst))


_BINARY_FN = {"add": "add", "iadd": "add", "sub": "sub", "isub": "sub", "mul": "mul", "imul": "mul",
              "truediv": "div", "div": "div", "maximum": "max", "minimum": "min"}
_ELTWISE_FN = {"relu": ("relu", 0.0, 0.0), "relu_": ("relu", 0.0, 0.0), "relu6": ("clip", 0.0, 6.0),
               "sigmoid": ("logistic", 0, 0), "tanh": ("tanh", 0, 0), "silu": ("swish", 1.0, 0.0),
               "hardswish": ("hardswish", 1 / 6, 0.5), "hardsigmoid": ("hardsigmoid", 1 / 6, 0.5),
               "mish": ("mish", 0, 0), "exp": ("exp", 0, 0), "sqrt": ("sqrt", 0, 0), "abs": ("abs", 0, 0),
               "neg": ("neg", 0, 0), "rsqrt": ("rsqrt", 0, 0), "erf": ("erf", 0, 0)}
_PY_OPS = {"floordiv": operator.floordiv, "mod": operator.mod, "eq": operator.eq, "ne": operator.ne,
           "lt": operator.lt, "le": operator.le, "gt": operator.gt, "ge": operator.ge, "add": operator.add,
           "sub": operator.sub, "mul": operator.mul, "truediv": operator.truediv, "neg": operator.neg,
           "and_": operator.and_, "or_": operator.or_, "not_": operator.not_}
_ALIAS_FN = {"flatten", "view", "reshape", "contiguous", "squeeze", "unsqueeze", "dropout", "clone", "detach",
             "stochastic_depth"}  # torchvision.ops.stochastic_depth: identity in eval


def _lower_call(L: _Lowerer, node, name, args, kw):
    """call_function / call_method -> primitives (or an alias, or a python value)."""
    g = L.g
    if name == "size":
        s = g.tensors[int(args[0])].shape
        return s[args[1]] if len(args) > 1 else s
    if name in ("dim", "ndimension") and isinstance(args[0], _Tid):
        return len(g.tensors[int(args[0])].shape)
    if name in ("getattr",) and args[1] == "shape":
        return g.tensors[int(args[0])].shape
    if name == "getitem" and not isinstance(args[0], _Tid):
        return args[0][args[1]]
    if name in ("_assert", "_assert_async"):   # a trace-time shape check (vit's class token): already true
        return None
    if name in _PY_OPS and not any(isinstance(a, _Tid) for a in list(args) + list(kw.values())):
        return _PY_OPS[name](*args)   # shape arithmetic on python values (bert's d // heads)
    if name in _ALIAS_FN:
        if not isinstance(args[0], _Tid) and hasattr(args[0], "reshape"):   # a constant (yolo's anchors): fold
            return args[0].reshape(L.shape(node))
        return _Tid(g.alias(int(args[0]), L.shape(node)))
    if name in _BINARY_FN:
        shape = L.shape(node)
        a, b = args[0], args[1]
        if not isinstance(a, _Tid) and not isinstance(b, _Tid):
            raise AiegraphError("dnn_torch", f"{node.name}: {name} of two constants")
        return _Tid(L.binary(node.name, _BINARY_FN[name], L.tid(a, len(shape)), L.tid(b, len(shape)), shape))
    if name in _ELTWISE_FN:
        alg, a, b = _ELTWISE_FN[name]
        return _Tid(L.eltwise(node.name, int(args[0]), alg, a, b))
    if name in ("gelu",):
        alg = "gelu_tanh" if kw.get("approximate", "none") == "tanh" else "gelu_erf"
        return _Tid(L.eltwise(node.name, int(args[0]), alg, 0.0, 0.0))
    if name in ("hardtanh", "clamp", "clip"):
        lo = kw.get("min", kw.get("min_val", args[1] if len(args) > 1 else -1.0))
        hi = kw.get("max", kw.get("max_val", args[2] if len(args) > 2 else 1.0))
        return _Tid(L.eltwise(node.name, int(args[0]), "clip", float(lo), float(hi)))
    if name in ("leaky_relu",):
        return _Tid(L.eltwise(node.name, int(args[0]), "relu", float(kw.get("negative_slope", args[1] if len(args) > 1 else 0.01)), 0.0))
    return _lower_call_tensor(L, node, name, args, kw)


def _slice(L: _Lowerer, name, x: int, starts, sizes, steps=None) -> int:
    """aiednn_slice: dst dims = sizes, element i of axis d read at starts[d] + i * steps[d]"""
    g = L.g
    y = g.new(tuple(sizes), g.tensors[x].dt)
    return g.op(name, "aiednn_slice_create", [("md", x), ("md", y), ("i64s", list(starts)),
                                              ("i64s", list(steps or [1] * len(sizes)))],
                [("SRC", x), ("DST", y)], y)


def _slice_getitem(L: _Lowerer, node, x: int, idx):
    """x[idx] with ints / slices / one Ellipsis"""
    shape = tuple(L.g.tensors[x].shape)
    idx = idx if isinstance(idx, tuple) else (idx,)
    if any(i is None for i in idx) or sum(i is Ellipsis for i in idx) > 1:
        raise AiegraphError("dnn_torch", f"{node.name}: getitem with None / several Ellipsis is not supported")
    if Ellipsis in idx:
        e = idx.index(Ellipsis)
        idx = idx[:e] + (slice(None),) * (len(shape) - len(idx) + 1) + idx[e + 1:]
    idx = idx + (slice(None),) * (len(shape) - len(idx))
    starts, sizes, steps = [], [], []
    for d, i in zip(shape, idx):
        if isinstance(i, int):
            starts.append(i % d), sizes.append(1), steps.append(1)
        elif isinstance(i, slice) and (i.step or 1) > 0:
            a, b, st = i.indices(d)
            starts.append(a), sizes.append(max(0, -(-(b - a) // st))), steps.append(st)
        else:
            raise AiegraphError("dnn_torch", f"{node.name}: getitem index {i!r} is not supported")
    if list(sizes) == list(shape):
        return _Tid(x)   # the whole tensor
    y = _slice(L, node.name, x, starts, sizes, steps)
    return _Tid(L.g.alias(y, L.shape(node)) if tuple(sizes) != tuple(L.shape(node)) else y)


def _split(L: _Lowerer, node, name, x: int, args, kw) -> tuple:
    """x.split / chunk -> one slice per piece (a tuple; the getitem nodes pick them)"""
    shape = tuple(L.g.tensors[x].shape)
    dim = kw.get("dim", args[2] if len(args) > 2 else 0) % len(shape)
    arg = kw.get("split_size_or_sections", kw.get("chunks", kw.get("split_sizes", args[1] if len(args) > 1 else None)))
    if name == "chunk":
        step = -(-shape[dim] // int(arg))
        sec = [min(step, shape[dim] - k) for k in range(0, shape[dim], step)]
    elif isinstance(arg, int):
        sec = [min(arg, shape[dim] - k) for k in range(0, shape[dim], arg)]
    else:
        sec = [int(v) for v in arg]
    out, off = [], 0
    for i, n in enumerate(sec):
        starts = [0] * len(shape)
        starts[dim] = off
        sizes = list(shape)
        sizes[dim] = n
        out.append(_Tid(_slice(L, f"{node.name}_{i}", x, starts, sizes)))
        off += n
    return tuple(out)


def _gqa_kv(node) -> bool:
    """x.repeat_interleave(r, dim=1) read only as an sdpa's K / V"""
    if not node.users:
        return False
    dim = node.kwargs.get("dim", node.args[2] if len(node.args) > 2 else None)
    if dim is None or len(node.meta["tensor_meta"].shape) != 4 or dim % 4 != 1:
        return False
    return all(_name_of(u) == "scaled_dot_product_attention" and u.args[0] is not node for u in node.users)


def _name_of(node) -> str:
    t = node.target
    return t if isinstance(t, str) else getattr(t, "__name__", str(t)).split(".")[-1]


def _lower_call_tensor(L: _Lowerer, node, name, args, kw):
    g = L.g
    if name == "getitem" and isinstance(args[0], _Tid):
        return _slice_getitem(L, node, int(args[0]), args[1])
    if name in ("split", "chunk", "split_with_sizes") and isinstance(args[0], _Tid):
        return _split(L, node, name, int(args[0]), args, kw)
    if name == "expand" and isinstance(args[0], _Tid) and tuple(g.tensors[int(args[0])].shape) == tuple(L.shape(node)):
        return args[0]   # broadcasts nothing (batch 1: vit's class token)
    if name == "expand" and hasattr(args[0], "expand"):   # a constant (vit's class-token parameter): fold it
        return args[0].expand(*[int(v) for v in (args[1] if len(args) == 2 and isinstance(args[1], (list, tuple))
                                                  else args[1:])]).contiguous()
    if name == "repeat_interleave" and isinstance(args[0], _Tid) and _gqa_kv(node):
        return args[0]   # the sdpa below reads the Hk heads itself
    if name == "pow" and isinstance(args[0], _Tid) and isinstance(kw.get("exponent", args[1] if len(args) > 1 else None),
                                                                  (int, float)):
        return _Tid(L.eltwise(node.name, int(args[0]), "pow", 1.0, float(kw.get("exponent", args[1]))))
    if name in ("cat", "concat"):
        rank = len(L.shape(node))
        xs = [int(t) if isinstance(t, _Tid) else L.tid(t, rank) for t in args[0]]   # constants (vit's class token)
        dim = kw.get("dim", args[1] if len(args) > 1 else 0)
        y = g.new(L.shape(node))
        dim %= len(g.tensors[y].shape)
        return _Tid(g.op(node.name, "aiednn_concat_create", [("int", len(xs)), ("mds", xs), ("int", dim), ("md", y)],
                         [(("multi", i), t) for i, t in enumerate(xs)] + [("DST", y)], y))
    if name in ("permute",):
        perm = args[1:] if not isinstance(args[1], (list, tuple)) else args[1]
        r = len(perm)
        return _Tid(L.reorder(node.name, int(args[0]), "f32", perm=[p % r for p in perm]))
    if name in ("transpose",):
        r = len(g.tensors[int(args[0])].shape)
        perm = list(range(r))
        a, b = args[1] % r, args[2] % r
        perm[a], perm[b] = perm[b], perm[a]
        return _Tid(L.reorder(node.name, int(args[0]), "f32", perm=perm))
    if name in ("mean",):
        x = int(args[0])
        r = len(g.tensors[x].shape)
        dims = kw.get("dim", args[1] if len(args) > 1 else list(range(r)))
        dims = [dims] if isinstance(dims, int) else list(dims)
        keep = list(g.tensors[x].shape)
        for d in dims:
            keep[d % r] = 1
        y = g.new(keep)
        g.op(node.name, "aiednn_reduction_create", [("alg", "reduction_mean"), ("float", 0.0), ("float", 0.0),
                                                    ("md", x), ("md", y)], [("SRC", x), ("DST", y)], y)
        return _Tid(g.alias(y, L.shape(node)))
    if name in ("max_pool2d", "_max_pool2d", "avg_pool2d"):
        def arg(i, key, dflt):
            return kw.get(key, args[i] if len(args) > i else dflt)
        k = arg(1, "kernel_size", None)
        st, pad = arg(2, "stride", None), arg(3, "padding", 0)
        if name == "avg_pool2d":
            if arg(6, "divisor_override", None):
                raise AiegraphError("dnn_torch", f"{node.name}: divisor_override not supported")
            alg = "avg_include_padding" if arg(5, "count_include_pad", True) else "avg_exclude_padding"
            return _Tid(L.pool(node, int(args[0]), alg, k, st, pad, 1, arg(4, "ceil_mode", False)))
        return _Tid(L.pool(node, int(args[0]), "max", k, st, pad, arg(4, "dilation", 1), arg(5, "ceil_mode", False)))
    if name in ("adaptive_avg_pool2d",):
        return _Tid(L.adaptive_avg(node, int(args[0])))
    if name in ("interpolate",):
        return _Tid(L.resample(node, int(args[0]), kw.get("mode", "nearest"), kw.get("align_corners")))
    if name in ("softmax", "log_softmax"):
        dim = kw.get("dim", args[1] if len(args) > 1 else -1)
        return _Tid(L.softmax(node, int(args[0]), dim, "log" if name == "log_softmax" else "accurate"))
    if name in ("matmul", "bmm", "matmul_"):
        a, b = int(args[0]), int(args[1])
        shape = L.shape(node)
        a, b = L.rank_to(a, len(shape)), L.rank_to(b, len(shape))
        y = g.new(shape)
        return _Tid(g.op(node.name, "aiednn_matmul_forward_create",
                         [("md", a), ("md", b), ("null",), ("md", y), ("int", 0)],
                         [("SRC", a), ("WEIGHTS", b), ("DST", y)], y))
    if name in ("linear",):
        w = args[1] if len(args) > 1 else kw["weight"]
        b = kw.get("bias", args[2] if len(args) > 2 else None)
        return _Tid(L.linear(node, int(args[0]), w.detach().numpy(),
                             None if b is None else b.detach().numpy()))
    if name in ("layer_norm",):
        w = kw.get("weight", args[2] if len(args) > 2 else None)
        b = kw.get("bias", args[3] if len(args) > 3 else None)
        eps = kw.get("eps", args[4] if len(args) > 4 else 1e-5)
        return _Tid(L._norm(node, "aiednn_layer_normalization_forward_inference_create", int(args[0]), eps, w, b))
    if name in ("scaled_dot_product_attention",):
        q, k, v = (int(t) for t in args[:3])
        mask = kw.get("attn_mask", args[3] if len(args) > 3 else None)
        if kw.get("dropout_p", args[4] if len(args) > 4 else 0.0):
            raise AiegraphError("dnn_torch", f"{node.name}: sdpa with dropout")
        if mask is not None and getattr(mask, "dtype", None) is not None and str(mask.dtype) == "torch.bool":
            raise AiegraphError("dnn_torch", f"{node.name}: sdpa with a boolean mask")
        m = None if mask is None else L.tid(mask, 4)   # additive float mask, broadcast to [B, H, Sq, Skv]
        D = g.tensors[q].shape[-1]
        scale = kw.get("scale", args[6] if len(args) > 6 else None) or 1.0 / math.sqrt(D)   # aten order
        y = g.new(L.shape(node))
        return _Tid(g.op(node.name, "aiednn_sdpa_forward_create",
                         [("md", q), ("md", k), ("md", v), ("null",) if m is None else ("md", m), ("md", y),
                          ("float", scale), ("int", int(bool(kw.get("is_causal", args[5] if len(args) > 5 else False))))],
                         [("SRC", q), ("SRC_1", k), ("SRC_2", v)] + ([("ATTN_MASK", m)] if m is not None else [])
                         + [("DST", y)], y))
    raise AiegraphError("dnn_torch", f"{node.name}: {name} is not supported")


def calibrate(gm, x) -> dict:
    """per conv / linear module: symmetric per-tensor input scale max|x| / 127"""
    import torch
    scales, hooks = {}, []
    for name, m in gm.named_modules():
        if isinstance(m, (torch.nn.Conv2d, torch.nn.ConvTranspose2d, torch.nn.Linear)):
            def hook(mod, inp, name=name):
                a = float(inp[0].detach().abs().max())
                scales[name] = max(scales.get(name, 0.0), a / 127.0 if a > 0 else 1.0)
            hooks.append(m.register_forward_pre_hook(hook))
    with torch.no_grad():
        gm(x)
    for h in hooks:
        h.remove()
    return scales


def prepare(model, x):
    """eval, fold Conv+BN, trace -> fx GraphModule"""
    import torch
    import torch.fx as fx
    from torch.fx.experimental.optimization import fuse
    model = _decompose_mha(model.eval())
    try:
        gm = fuse(model)
    except Exception:  # noqa: BLE001 -- fuse traces too; fall back to a plain trace
        try:
            gm = fx.symbolic_trace(model)
        except Exception:  # noqa: BLE001 -- python control flow over modules (ultralytics' layer loop):
            gm = _dynamo_trace(model, x)  # Dynamo traces through it, specialized to x's shapes
    with torch.no_grad():
        gm(x)
    return gm


def mha_module(m):
    """nn.MultiheadAttention *m* (batch-first self-attention, packed in_proj) as modules the lowerings know"""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class _MHA(nn.Module):
        def __init__(self, m):
            super().__init__()
            e = m.embed_dim
            self.h, self.e = m.num_heads, e
            self.qkv = nn.Linear(e, 3 * e, bias=m.in_proj_bias is not None)
            self.out = nn.Linear(e, e, bias=m.out_proj.bias is not None)
            with torch.no_grad():
                self.qkv.weight.copy_(m.in_proj_weight)
                if m.in_proj_bias is not None:
                    self.qkv.bias.copy_(m.in_proj_bias)
                self.out.weight.copy_(m.out_proj.weight)
                if m.out_proj.bias is not None:
                    self.out.bias.copy_(m.out_proj.bias)

        def forward(self, q, k, v, need_weights=False, attn_mask=None, key_padding_mask=None):
            b, s, _ = q.shape
            t = self.qkv(q)   # self-attention: k and v are q (checked at the swap: torchvision passes x, x, x)
            qq, kk, vv = t.split(self.e, dim=2)
            hs = lambda z: z.reshape(b, s, self.h, self.e // self.h).transpose(1, 2)
            a = F.scaled_dot_product_attention(hs(qq), hs(kk), hs(vv))
            return self.out(a.transpose(1, 2).reshape(b, s, self.e)), None

    return _MHA(m)


def _decompose_mha(model):
    """Swap nn.MultiheadAttention for the equivalent modules the lowering knows."""
    import copy

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    if not any(isinstance(m, nn.MultiheadAttention) for m in model.modules()):
        return model

    model = copy.deepcopy(model)
    for name, m in list(model.named_modules()):
        if isinstance(m, nn.MultiheadAttention):
            if not (m.batch_first and m._qkv_same_embed_dim):   # eval: dropout is off
                raise AiegraphError("dnn_torch", f"{name}: MultiheadAttention form not supported")
            parent = model.get_submodule(name.rsplit(".", 1)[0]) if "." in name else model
            setattr(parent, name.rsplit(".", 1)[-1], mha_module(m))
    return model


def _dynamo_trace(model, x):
    """torch._dynamo.export as the tracer of last resort, Conv+BN folded after it."""
    import torch
    from torch.fx.experimental.optimization import fuse
    with torch.no_grad():
        model(x)
        gm = torch._dynamo.export(model)(x).graph_module
    try:
        return fuse(gm, no_trace=True)
    except Exception:  # noqa: BLE001 -- no Conv+BN pattern fuse understands: keep it unfused
        return gm


def lower(model, x, mode: str = "f32", name: str = "model", gm=None, qscale=None, fuse: bool = True) -> AGraph:
    """Lower *model* (traced on example input *x*, shape [1,...]) to aiednn primitives."""
    if mode not in ("f32", "int8"):
        raise AiegraphError("dnn_torch", f"mode={mode!r}", hint="'f32' or 'int8'")
    gm = gm if gm is not None else prepare(model, x)
    if mode == "int8" and qscale is None:
        qscale = calibrate(gm, x)
    lw = _Lowerer(gm, mode, qscale or {})
    lw.fuse = fuse
    g = lw.run(x)
    g.name = name
    g.tensors[g.input].data = x.detach().numpy().astype(np.float32)
    return g


def _quantize_residuals(gm, qscale: dict):
    """the int8 skip streams of --aiednn-quantize-residual, in the fx graph"""
    import operator

    import torch
    mods = dict(gm.named_modules())
    gemm = (torch.nn.Conv2d, torch.nn.ConvTranspose2d, torch.nn.Linear)
    is_gemm = lambda n: n.op == "call_module" and isinstance(mods.get(n.target), gemm)  # noqa: E731

    def through(n):  # eval-time identities the lowering drops (Dropout, StochasticDepth, Identity)
        while (((n.op == "call_module" and type(mods.get(n.target)).__name__ in ("Dropout", "StochasticDepth", "Identity"))
                or (n.op == "call_function" and getattr(n.target, "__name__", "") in ("stochastic_depth", "dropout")))
               and len(n.users) == 1 and isinstance(n.args[0], torch.fx.Node)):
            n = n.args[0]
        return n
    adds = [n for n in gm.graph.nodes if (n.op == "call_function" and n.target in (operator.add, operator.iadd, torch.add))
            or (n.op == "call_method" and n.target in ("add", "add_"))]
    for n in adds:
        a = [v for v in n.args[:2] if isinstance(v, torch.fx.Node)]
        if len(a) != 2 or a[0] is a[1]:
            continue
        for skip, other in ((a[0], a[1]), (a[1], a[0])):
            q = next((u for u in skip.users if is_gemm(u) and u.target in qscale), None)
            if q is None and len(skip.users) == 1:
                # the skip feeds nothing quantized: use the block output's scale
                out = n
                if len(out.users) == 1:
                    u = next(iter(out.users))
                    if (u.op == "call_module" and type(mods.get(u.target)).__name__ == "ReLU") or \
                            (u.op == "call_function" and getattr(u.target, "__name__", "") == "relu"):
                        out = u
                q = next((u for u in out.users if is_gemm(u) and u.target in qscale), None)
            other = through(other)
            if not is_gemm(other) or len(other.users) != 1 or q is None:
                continue
            with gm.graph.inserting_before(n):
                fq = gm.graph.call_function(_fake_q8, (skip, float(qscale[q.target])))
            n.replace_input_with(skip, fq)
            break
    gm.recompile()


def _fake_q8(t, s: float):
    import torch
    return torch.clamp(torch.round(t / s), -128, 127) * s


def fake_quant_reference(gm, x, qscale: dict, in_place: bool = False, quant_residual: bool = False):
    """PyTorch with the int8 mode's quantization simulated"""
    import copy

    import torch
    ref = gm if in_place else copy.deepcopy(gm)
    if quant_residual:
        _quantize_residuals(ref, qscale)
    hooks = []
    for name, m in ref.named_modules():
        if not isinstance(m, (torch.nn.Conv2d, torch.nn.ConvTranspose2d, torch.nn.Linear)):
            continue
        w = m.weight.data
        if isinstance(m, torch.nn.ConvTranspose2d):     # per OUTPUT channel: dim 1 within each group
            g, Cg = m.groups, w.shape[0] // m.groups
            wq = w.clone()
            for i in range(g):
                blk = w[i * Cg:(i + 1) * Cg]
                s = blk.abs().amax(dim=(0, 2, 3), keepdim=True) / 127.0
                s[s == 0] = 1.0
                wq[i * Cg:(i + 1) * Cg] = torch.clamp(torch.round(blk / s), -127, 127) * s
        else:
            s = w.abs().amax(dim=tuple(range(1, w.ndim)), keepdim=True) / 127.0
            s[s == 0] = 1.0
            wq = torch.clamp(torch.round(w / s), -127, 127) * s
        m.weight.data = wq
        sx = qscale[name]

        def hook(mod, inp, sx=sx):
            return (torch.clamp(torch.round(inp[0] / sx), -128, 127) * sx,) + tuple(inp[1:])
        hooks.append(m.register_forward_pre_hook(hook))
    with torch.no_grad():
        y = ref(x)
    return y[0] if isinstance(y, (list, tuple)) else y


# ── x86 execution through ctypes ──────────────────────────────────────────

def _bufs(g: AGraph) -> dict:
    roots = {}
    arena = np.zeros(g.arena, np.uint8) if g.plan is not None else None
    for t in g.tensors:
        if t.base is None:
            if arena is not None and t.id in g.plan:     # the MLIR memory plan, in-place ops included
                n = int(np.prod(t.shape)) * np.dtype(DT_NP[t.dt]).itemsize
                roots[t.id] = arena[g.plan[t.id]:g.plan[t.id] + n].view(DT_NP[t.dt]).reshape(t.shape)
                continue
            roots[t.id] = t.data if t.data is not None else np.zeros(t.shape, DT_NP[t.dt])
    out = {}
    for t in g.tensors:
        r = g.root(t.id)
        out[t.id] = roots[r].reshape(t.shape)
    return out


def run_ctypes(g: AGraph, lib, x=None, keep: bool = False):
    """Execute *g* on x86 (``dnn_ctypes.AiednnLib``)."""
    import ctypes as C

    from .dnn_ctypes import ARG, MULTIPLE_SRC, ConvParams, PoolParams, md
    if x is not None:
        g.tensors[g.input].data = np.ascontiguousarray(x, np.float32)
    bufs = _bufs(g)
    impls = []
    for op in g.ops:
        keepalive, cargs = [], []
        for item in op.create:
            k = item[0]
            if k == "md":
                m = md(g.tensors[item[1]].shape, g.tensors[item[1]].dt)
                keepalive.append(m)
                cargs.append(C.byref(m))
            elif k == "null":
                cargs.append(None)
            elif k == "int":
                cargs.append(C.c_int(item[1]))
            elif k == "i64":
                cargs.append(C.c_int64(item[1]))
            elif k == "uint":
                cargs.append(C.c_uint(item[1]))
            elif k == "float":
                cargs.append(C.c_float(item[1]))
            elif k == "alg":
                cargs.append(C.c_int(lib.ALG[item[1]]))
            elif k in ("conv", "pool"):
                p = ConvParams() if k == "conv" else PoolParams()
                d = item[1]
                for i in range(2):
                    if k == "pool":
                        p.kernel[i] = d["kernel"][i]
                    p.strides[i], p.padding_l[i], p.padding_r[i], p.dilates[i] = (
                        d["strides"][i], d["pl"][i], d["pr"][i], d["dil"][i])
                if k == "conv":
                    p.groups = d["groups"]
                keepalive.append(p)
                cargs.append(C.byref(p))
            elif k == "perm":
                a = (C.c_int * len(item[1]))(*item[1])
                keepalive.append(a)
                cargs.append(a)
            elif k == "i64s":
                a = (C.c_int64 * len(item[1]))(*item[1])
                keepalive.append(a)
                cargs.append(a)
            elif k == "mds":
                arr = (type(md((1,), "f32")) * len(item[1]))(*[md(g.tensors[t].shape, g.tensors[t].dt) for t in item[1]])
                keepalive.append(arr)
                cargs.append(arr)
            elif k == "attr":
                from .dnn_ctypes import attr as mk_attr
                post = [po if po[0] == "eltwise" else ("binary", po[1], g.tensors[po[2]].shape, g.tensors[po[2]].dt)
                        + tuple(po[3:5]) for po in item[1]["post"]]
                a, held = mk_attr(lib.ALG, scales=item[1]["scales"], post_ops=post,
                                  dst_scale=item[1].get("dst_scale", 0.0), dst_zp=item[1].get("dst_zp", 0),
                                  flags=item[1].get("flags", 0))
                keepalive += [a, held]
                cargs.append(C.byref(a))
            else:
                raise AiegraphError("dnn_torch", f"{op.name}: create item {k}")
        prim, st = lib.create(op.fn, *cargs)
        if st != 0:
            raise AiegraphError("dnn_torch", f"{op.name}: {op.fn}: {lib.status(st)}")
        impls.append(lib.impl(prim))
        ex = {}
        for a, t in op.exec:
            if isinstance(a, tuple):
                ex[(128 if a[0] == "post" else MULTIPLE_SRC) + a[1]] = bufs[t]
            else:
                ex[ARG[a.lower()]] = bufs[t]
        st = lib.execute(prim, ex)
        lib.destroy(prim)
        if st != 0:
            raise AiegraphError("dnn_torch", f"{op.name}: execute: {lib.status(st)}")
    out = bufs[g.output].copy()
    return (out, impls, bufs) if keep else (out, impls)


# ── C app emission ────────────────────────────────────────────────────────

def _plan_memory(g: AGraph) -> tuple:
    """arena offsets for activation roots, reusing buffers after their last use"""
    roots = [t for t in g.tensors if t.base is None and t.kind == "act"]
    first, last = {}, {}
    for i, op in enumerate(g.ops):
        for _, t in op.exec:
            r = g.root(t)
            first.setdefault(r, i)
            last[r] = i
    last[g.root(g.output)] = len(g.ops)
    live, offs, top = [], {}, 0
    for t in sorted(roots, key=lambda t: first.get(t.id, 0)):
        if t.id not in first:
            continue
        start = first[t.id]
        live = [(o, s, e) for o, s, e in live if e >= start]
        size = (int(np.prod(t.shape)) * np.dtype(DT_NP[t.dt]).itemsize + 63) // 64 * 64
        off = 0
        for o, s, _ in sorted(live):
            if off + size <= o:
                break
            off = max(off, o + s)
        offs[t.id] = off
        live.append((off, size, last[t.id]))
        top = max(top, off + size)
    return offs, top


def _c_md(t: Tensor) -> str:
    return f"{{{len(t.shape)}, {{{', '.join(map(str, t.shape))}}}, {DT_ENUM[t.dt]}}}"


def _int_gemm(g: AGraph, op: Op) -> bool:
    return op.fn in _GEMM_FNS and g.tensors[op.create[0][1]].dt != "f32"


def emit_app(g: AGraph, out: Path, golden: np.ndarray, tol: float, float_top1: int = -1,
             verify_gemm: bool = True, autotune=None, prefer: str = "aie") -> Path:
    """C app running *g* through aiednn on the x86 / board build, checking the output against *golden*"""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    blob, coff = bytearray(), {}
    for t in g.tensors:
        if t.base is None and t.kind in ("const", "input"):
            coff[t.id] = len(blob)
            blob += np.ascontiguousarray(t.data, DT_NP[t.dt]).tobytes()
            blob += b"\0" * (-len(blob) % 64)
    gold_off = len(blob)
    blob += np.ascontiguousarray(golden, np.float32).tobytes()
    (out / f"{g.name}_blob.bin").write_bytes(bytes(blob))
    aoff, arena = (g.plan, g.arena) if g.plan is not None else _plan_memory(g)
    check = max([int(np.prod(g.tensors[op.out].shape)) * 4 for op in g.ops if _int_gemm(g, op)] + [64])
    lines = _app_head(g, out, arena, gold_off, int(np.prod(golden.shape)), tol, float_top1)
    lines.insert(lines.index(f"#define FLOAT_TOP1 {float_top1}"),   # images: one top-1 per row of [batch, classes]
                 f"#define BATCH {golden.shape[0] if golden.ndim >= 2 else 1}")
    lines += [f"#define VERIFY_GEMM {int(verify_gemm)}",
              f"static unsigned char g_check[{check}] __attribute__((aligned(64)));",
              "static int g_gemm_checked, g_gemm_bad;", ""]
    for t in g.tensors:
        r = g.root(t.id)
        base = f"g_blob + {coff[r]}" if r in coff else f"g_arena + {aoff.get(r, 0)}"
        lines.append(f"#define P{t.id} ((void *)({base}))")
    lines += ["", "static int run_net(void) {", "    aiednn_status_t st;", "    aiednn_primitive_t p;"]
    for i, op in enumerate(g.ops):
        lines += _c_op(g, i, op, _int_gemm(g, op))
    lines += ["    return 0;", "}", ""] + _app_main(g, autotune, prefer)
    path = out / f"{g.name}_app.cc"
    path.write_text("\n".join(lines) + "\n")
    return path


def _app_head(g, out, arena, gold_off, ngold, tol, float_top1) -> list:
    return [
        f"/* Generated by aiegraph/dnn_torch.py: {g.summary()} */",
        '#include "aiednn.h"', "#include <math.h>", "#include <stdio.h>", "#include <string.h>", "",
        "#define INCBIN(sym, path) \\",
        '    __asm__(".section .rodata\\n.balign 64\\n.global " #sym "\\n" #sym ":\\n.incbin \\"" path "\\"\\n.previous\\n")',
        f'INCBIN(g_blob_data, "{(out / f"{g.name}_blob.bin").resolve()}");',
        'extern "C" const unsigned char g_blob_data[];',
        "#define g_blob ((unsigned char *)g_blob_data)",
        f"static unsigned char g_arena[{max(arena, 64)}] __attribute__((aligned(64)));",
        f"#define GOLD_OFF {gold_off}", f"#define NGOLD {ngold}", f"#define TOL {tol!r}",
        f"#define FLOAT_TOP1 {float_top1}", f'#define MODEL_NAME "{g.name}"', f'#define MODE "{g.mode}"',
        f"#define NOPS {len(g.ops)}", "",
        "#if defined(__aarch64__)",
        "static uint64_t now_ticks(void) { uint64_t v; __asm__ volatile(\"isb; mrs %0, cntvct_el0\" : \"=r\"(v)); return v; }",
        "static double ticks_ms(uint64_t t) { uint64_t f; __asm__ volatile(\"mrs %0, cntfrq_el0\" : \"=r\"(f)); return 1e3 * (double)t / (double)f; }",
        "#else", "#include <time.h>",
        "static uint64_t now_ticks(void) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec; }",
        "static double ticks_ms(uint64_t t) { return (double)t / 1e6; }", "#endif",
        "static uint64_t g_op_ticks[NOPS];", "static int g_warmup; /* run 1 of 2: no per-layer check */", "static const char *g_op_impl[NOPS];", "static char g_impl_buf[NOPS][64];",
        "static const char *g_op_name[NOPS];", "",
        "static int fail(int i, const char *what, aiednn_status_t st) {",
        '    printf("FAIL: op %d (%s) %s: %s\\n", i, g_op_name[i], what, aiednn_status_str(st));',
        "    return 1;", "}", ""]


def _c_op(g: AGraph, i: int, op: Op, int_gemm: bool = False) -> list:
    pre, cargs = [], []
    for j, item in enumerate(op.create):
        k = item[0]
        if k == "md":
            pre.append(f"        static const aiednn_memory_desc_t m{j} = {_c_md(g.tensors[item[1]])};")
            cargs.append(f"&m{j}")
        elif k == "null":
            cargs.append("nullptr")
        elif k in ("int", "i64", "uint"):
            cargs.append(str(int(item[1])) + ("u" if k == "uint" else ""))
        elif k == "float":
            cargs.append(f"{float(item[1])!r}f".replace("inf", "INFINITY"))
        elif k == "alg":
            cargs.append(f"aiednn_{item[1]}")
        elif k == "conv":
            d = item[1]
            pre.append(f"        static const aiednn_conv_params_t c{j} = {{{{{d['strides'][0]}, {d['strides'][1]}}}, "
                       f"{{{d['pl'][0]}, {d['pl'][1]}}}, {{{d['pr'][0]}, {d['pr'][1]}}}, "
                       f"{{{d['dil'][0]}, {d['dil'][1]}}}, {d['groups']}}};")
            cargs.append(f"&c{j}")
        elif k == "pool":
            d = item[1]
            pre.append(f"        static const aiednn_pool_params_t c{j} = {{{{{d['kernel'][0]}, {d['kernel'][1]}}}, "
                       f"{{{d['strides'][0]}, {d['strides'][1]}}}, {{{d['pl'][0]}, {d['pl'][1]}}}, "
                       f"{{{d['pr'][0]}, {d['pr'][1]}}}, {{{d['dil'][0]}, {d['dil'][1]}}}}};")
            cargs.append(f"&c{j}")
        elif k == "perm":
            pre.append(f"        static const int c{j}[] = {{{', '.join(map(str, item[1]))}}};")
            cargs.append(f"c{j}")
        elif k == "i64s":
            pre.append(f"        static const int64_t c{j}[] = {{{', '.join(map(str, item[1]))}}};")
            cargs.append(f"c{j}")
        elif k == "mds":
            pre.append(f"        static const aiednn_memory_desc_t c{j}[] = {{"
                       + ", ".join(_c_md(g.tensors[t]) for t in item[1]) + "};")
            cargs.append(f"c{j}")
        elif k == "attr":
            d = item[1]
            sc = d["scales"]
            if sc is not None:
                pre.append(f"        static const float s{j}[] = {{{', '.join(f'{float(v)!r}f' for v in sc)}}};")
            pos = []
            for po in d["post"]:
                if po[0] == "eltwise":
                    pos.append(f"{{aiednn_post_op_eltwise, aiednn_eltwise_{po[1]}, {float(po[2])!r}f, {float(po[3])!r}f, {{}}}}")
                else:
                    rs, rz = (float(po[3]), float(po[4])) if len(po) > 3 else (0.0, 0.0)  # s8 src1 scale / zp
                    pos.append(f"{{aiednn_post_op_binary, aiednn_binary_{po[1]}, {rs!r}f, {rz!r}f, "
                               f"{_c_md(g.tensors[po[2]])}}}")
            pre.append(f"        static {'' if d.get('flags') else 'const '}aiednn_attr_t c{j} = "
                       f"{{{len(sc) if sc is not None else 0}, "
                       f"{f's{j}' if sc is not None else 'nullptr'}, {float(d.get('dst_scale', 0.0))!r}f, "
                       f"{int(d.get('dst_zp', 0))}, {len(pos)}, {{{', '.join(pos)}}}}};")
            if d.get("flags"):     # AIEDNN_ATTR_* opt-ins: the last field, past the positional initializer
                pre.append(f"        c{j}.flags = {int(d['flags'])}u;")
            cargs.append(f"&c{j}")
        elif k == "floats":
            pre.append(f"        static const float c{j}[] = {{{', '.join(f'{v!r}f' for v in item[1])}}};")
            cargs.append(f"c{j}")
    ex = ", ".join((f"{{AIEDNN_ARG_ATTR_POST_OP_SRC({a[1]}), P{t}}}" if a[0] == "post" else
                    f"{{AIEDNN_ARG_MULTIPLE_SRC + {a[1]}, P{t}}}") if isinstance(a, tuple) else f"{{AIEDNN_ARG_{a}, P{t}}}"
                   for a, t in op.exec)
    return (["    {", f'        g_op_name[{i}] = "{op.name}";'] + pre +
            [f"        st = {op.fn}(&p, {', '.join(cargs)});",
             f'        if (st != aiednn_success) return fail({i}, "create", st);']
            + (["        aiednn_primitive_set_constant_weights(p, 1); // model weights: packed blocks cached"]
               if _const_weights(g, op) else []) +
            [f"        snprintf(g_impl_buf[{i}], 64, \"%s\", aiednn_primitive_impl_info(p));",
             f"        g_op_impl[{i}] = g_impl_buf[{i}];",
             f"        const aiednn_exec_arg_t a[] = {{{ex}}};",
             f"        aiednn_op_begin({i}); // per-op usage; a multi-megakernel build switches here",
             "        const uint64_t t0 = now_ticks();",
             f"        st = aiednn_primitive_execute_args(p, {len(op.exec)}, a);",
             f"        g_op_ticks[{i}] += now_ticks() - t0;",
             "        aiednn_primitive_destroy(p);",
             f'        if (st != aiednn_success) return fail({i}, "execute", st);']
            + (_c_verify(g, i, op, cargs, ex) if int_gemm else []) + ["    }"])


def _const_weights(g: AGraph, op: Op) -> bool:
    """a GEMM-family op whose WEIGHTS are model constants (in the blob), not an activation"""
    w = next((t for a, t in op.exec if a == "WEIGHTS"), None)
    return op.fn in _GEMM_FNS and w is not None and g.tensors[g.root(w)].data is not None and g.root(w) != g.input


def _c_verify(g: AGraph, i: int, op: Op, cargs: list, ex: str) -> list:
    """same primitive forced to ref:, same inputs, DST -> g_check; compare bytes"""
    dst = g.tensors[op.out]
    nbytes = int(np.prod(dst.shape)) * np.dtype(DT_NP[dst.dt]).itemsize
    exv = ex.replace(f"{{AIEDNN_ARG_DST, P{op.out}}}", "{AIEDNN_ARG_DST, g_check}")
    return ["#if VERIFY_GEMM",
            "        if (!g_warmup && strncmp(g_op_impl[%d], \"ref:\", 4)) {" % i,
            '            aiednn_set_impl_filter("ref:");',
            f"            st = {op.fn}(&p, {', '.join(cargs)});",
            "            aiednn_set_impl_filter(nullptr);",
            f'            if (st != aiednn_success) return fail({i}, "create ref", st);',
            f"            const aiednn_exec_arg_t r[] = {{{exv}}};",
            f"            st = aiednn_primitive_execute_args(p, {len(op.exec)}, r);",
            "            aiednn_primitive_destroy(p);",
            f'            if (st != aiednn_success) return fail({i}, "execute ref", st);',
            f"            const int bad = memcmp(g_check, P{op.out}, {nbytes}) != 0;",
            "            g_gemm_checked++;",
            "            g_gemm_bad += bad;",
            "            if (bad) {",
            f"                const unsigned char *x = (const unsigned char *)P{op.out}, *y = (const unsigned char *)g_check;",
            f"                long n = 0, first = -1;",
            f"                for (long k = 0; k < {nbytes}; k++)",
            "                    if (x[k] != y[k]) { n++; if (first < 0) first = k; }",
            f'                printf("GEMMCHK MISMATCH op {i} ({op.name}) %s vs ref: %ld of {nbytes} bytes differ, '
            f'first at byte %ld (%u vs ref %u)\\n", g_op_impl[{i}], n, first, x[first], y[first]);',
            "            }",
            "        }", "#endif"]


def _autotune_lines(autotune) -> list:
    """AIEDNN_TUNE setup lines for *autotune*: None, "tune" or a saved table's text."""
    if not autotune:
        return []
    if autotune == "tune":
        return ["    aiednn_set_autotune(1); /* run 1 times every implementation that accepts; run 2 uses the fastest */"]
    lit = "".join(f"\"{line}\\n\"\n        " for line in autotune.strip().splitlines() if line.startswith("AIEDNN_TUNE"))
    return [f"    static const char *tuned =\n        {lit.rstrip()};",
            '    printf("autotune: %d choices loaded\\n", aiednn_autotune_load(tuned));']


def _app_main(g: AGraph, autotune=None, prefer: str = "aie") -> list:
    lo, hi, grp = g.top1_range or (0, 1 << 30, 1)   # top-1 window of each image's output (a detector: its class rows)
    return [
        'extern "C" long aiednn_megakernel_switches __attribute__((weak)); /* gemm_uk.cc: ELF reloads */',
        "int main() {",
        '    printf("=== aiednn model " MODEL_NAME " [" MODE "] ===\\n");', *_autotune_lines(autotune),
        *(["    aiednn_set_preference(aiednn_prefer_fastest); /* cost refusals: small ops on ref: when faster */"]
          if prefer == "fastest" else []),
        "    /* run 1 (cold, untimed for time_ms, no per-layer check): one-time work -- weight reorders and",
        "     * packed weight blocks (constant weights are cached), the AIE ELF load and partition setup;",
        "     * run 2 is the steady state: time_ms, the per-layer check and every stat are from it */",
        "    g_warmup = 1;",
        "    if (run_net()) return 1;",
        "    double cold = 0;",
        "    for (int i = 0; i < NOPS; i++) cold += ticks_ms(g_op_ticks[i]), g_op_ticks[i] = 0;",
        "    aiednn_stats_reset();",
        "    g_warmup = 0;",
        "    const long sw0 = &aiednn_megakernel_switches ? aiednn_megakernel_switches : 0;",
        "    if (run_net()) return 1;",
        '    printf("MEGAKERNEL switches=%ld in the timed run\\n", (&aiednn_megakernel_switches ? aiednn_megakernel_switches : 0) - sw0);',
        "    double ms = 0;  /* primitive executions only: no create / destroy / VERIFY_GEMM re-runs */",
        "    for (int i = 0; i < NOPS; i++) ms += ticks_ms(g_op_ticks[i]);",
        '    printf("MODEL_COLD time_ms=%.1f (first run: one-time weight packing, ELF load)\\n", cold);',
        f"    const float *y = (const float *)P{g.output}, *ref = (const float *)(g_blob + GOLD_OFF);",
        "    double num = 0, den = 0; int top = 0, rtop = 0, miss = 0;",
        "    for (int i = 0; i < NGOLD; i++)",
        "        num += ((double)y[i] - ref[i]) * ((double)y[i] - ref[i]), den += (double)ref[i] * ref[i];",
        "    for (int b = 0; b < BATCH; b++) { /* top-1 per image; top / rtop: image 0's */",
        f"        const int per = NGOLD / BATCH, lo = {lo}, hi = {hi} < per ? {hi} : per; int t = lo, r = lo;",
        "        for (int i = lo + 1; i < hi; i++) {",
        "            if (y[b * per + i] > y[b * per + t]) t = i;", "            if (ref[b * per + i] > ref[b * per + r]) r = i;", "        }",
        f"        t = (t - lo) / {grp}, r = (r - lo) / {grp};",
        "        if (b == 0) top = t, rtop = r;", "        miss += t != r;", "    }",
        "    const double rel = sqrt(num / (den > 0 ? den : 1));",
        "    /* per-implementation time: where the inference went */",
        "    static const char *seen[NOPS]; static double tms[NOPS]; static int cnt[NOPS]; int ns = 0;",
        "    for (int i = 0; i < NOPS; i++) {", "        int k = 0;",
        "        while (k < ns && strcmp(seen[k], g_op_impl[i])) k++;",
        "        if (k == ns) seen[ns++] = g_op_impl[i];",
        "        tms[k] += ticks_ms(g_op_ticks[i]); cnt[k]++;", "    }",
        "    for (int k = 0; k < ns; k++)",
        '        printf("PROF %-40s %4d ops %9.2f ms\\n", seen[k], cnt[k], tms[k]);',
        "    aiednn_print_stats();",
        *(["    aiednn_autotune_dump();"] if autotune == "tune" else []),
        '    printf("MODEL " MODEL_NAME " mode=" MODE " ops=%d time_ms=%.1f relerr=%.3e top1=%d ref_top1=%d float_top1=%d"',
        '           " gemm_checked=%d gemm_bad=%d batch=%d top1_miss=%d ms_per_image=%.2f\\n", NOPS, ms, rel, top, rtop,',
        "           FLOAT_TOP1, g_gemm_checked, g_gemm_bad, BATCH, miss, ms / BATCH);",
        "    if (!(rel < TOL) || miss || g_gemm_bad) {",
        '        printf("FAIL: aiednn model " MODEL_NAME " differs from its reference (rel %.3e, tol %g)\\n", rel, TOL);',
        "        return 1;", "    }",
        '    printf("PASS: aiednn model " MODEL_NAME " [" MODE "] matches PyTorch, rel %.3e, top-1 %d\\n", rel, top);',
        "    return 0;", "}"]


# ── per-layer comparison ──────────────────────────────────────────────────

def layer_diff(model, x, mode: str, lib, top: int = 10) -> list:
    """[(node, rel error, shape)] of the graph vs the fake-quant PyTorch model, in graph order."""
    import torch
    from torch.fx import Interpreter
    gm = prepare(model, x)
    qscale = calibrate(gm, x) if mode == "int8" else None
    g = lower(model, x, mode, gm=gm, qscale=qscale)
    _, _, bufs = run_ctypes(g, lib, x.numpy(), keep=True)
    ref = gm
    if mode == "int8":
        import copy
        ref = copy.deepcopy(gm)
        fake_quant_reference(ref, x, qscale, in_place=True)
    seen = {}

    class Rec(Interpreter):
        def run_node(self, n):
            r = super().run_node(n)
            if isinstance(r, torch.Tensor):
                seen[n.name] = r.detach().numpy().copy()  # in-place ops (ReLU6) would rewrite a view
            return r
    with torch.no_grad():
        Rec(ref).run(x)
    # a fused op's buffer holds its LAST absorbed node (conv + silu -> silu's value):
    # compare each buffer only against the latest node, in execution order, that maps to it
    order = {name: i for i, name in enumerate(seen)}
    last = {}
    for name, tid in g.nodes.items():
        if name in order and (tid not in last or order[name] > order[last[tid]]):
            last[tid] = name
    keep = set(last.values())
    out = []
    for name, tid in g.nodes.items():
        if name in keep and seen[name].shape == bufs[tid].shape:
            a, b = bufs[tid].astype(np.float64), seen[name].astype(np.float64)
            out.append((name, float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30)), b.shape))
    return out


# ── layer-local exactness of the int8 GEMM family ─────────────────────────

_GEMM_FNS = ("aiednn_convolution_forward_create", "aiednn_deconvolution_forward_create",
             "aiednn_matmul_forward_create", "aiednn_convolution_forward_create_attr",
             "aiednn_deconvolution_forward_create_attr", "aiednn_matmul_forward_create_attr")


def _post_eltwise(alg, v, a, b):
    import torch
    import torch.nn.functional as F
    f = {"relu": lambda x: torch.where(x > 0, x, a * x), "clip": lambda x: x.clamp(a, b),
         "linear": lambda x: a * x + b, "logistic": torch.sigmoid, "tanh": torch.tanh,
         "swish": lambda x: x * torch.sigmoid(a * x), "elu": lambda x: torch.where(x > 0, x, a * (torch.exp(x) - 1)),
         "gelu_erf": F.gelu, "gelu_tanh": lambda x: F.gelu(x, approximate="tanh"),
         "hardswish": lambda x: x * (a * x + b).clamp(0, 1), "hardsigmoid": lambda x: (a * x + b).clamp(0, 1),
         "mish": F.mish, "soft_relu": lambda x: F.softplus(a * x) / a}
    return f[alg](v)


def gemm_exact_check(g: AGraph, bufs: dict) -> tuple:
    """Recompute every integer GEMM-like primitive in PyTorch from its own inputs; (checked, [(op, mismatches)])."""
    import torch
    import torch.nn.functional as F
    checked, bad = 0, []
    for op in g.ops:
        if op.fn not in _GEMM_FNS:
            continue
        mds = [g.tensors[c[1]] for c in op.create if c[0] == "md"]
        if mds[0].dt == "f32":
            continue
        ex = dict((a, t) for a, t in op.exec if not isinstance(a, tuple))
        x = torch.from_numpy(bufs[ex["SRC"]].astype(np.float64))
        w = torch.from_numpy(bufs[ex["WEIGHTS"]].astype(np.float64))
        if "matmul" in op.fn:
            tr = [c[1] for c in op.create if c[0] == "int"][-1]
            ref = torch.matmul(x, w.transpose(-1, -2) if tr else w)
        else:
            cp = next(c[1] for c in op.create if c[0] == "conv")
            dil = (cp["dil"][0] + 1, cp["dil"][1] + 1)
            if op.fn.startswith("aiednn_convolution"):
                xp = F.pad(x, (cp["pl"][1], cp["pr"][1], cp["pl"][0], cp["pr"][0]))
                ref = F.conv2d(xp, w, stride=cp["strides"], dilation=dil, groups=cp["groups"])
            else:
                G, O = cp["groups"], w.shape[0]
                Og, Cg = O // G, w.shape[1]
                wt = torch.cat([w[i * Og:(i + 1) * Og].transpose(0, 1) for i in range(G)], 0)  # [C, Og, kh, kw]
                full = F.conv_transpose2d(x, wt, stride=cp["strides"], dilation=dil, groups=G)
                Ho, Wo = bufs[ex["DST"]].shape[2:]
                ref = full[:, :, cp["pl"][0]:cp["pl"][0] + Ho, cp["pl"][1]:cp["pl"][1] + Wo]
        attr = next((c[1] for c in op.create if c[0] == "attr"), None)
        dst = bufs[ex["DST"]]
        ch = 1 if "convolution" in op.fn else dst.ndim - 1
        if attr and attr["scales"] is not None:
            sh = [1] * dst.ndim
            sh[ch] = -1
            ref = ref * torch.from_numpy(np.asarray(attr["scales"], np.float32).astype(np.float64)).reshape(sh)
        if "BIAS" in ex:
            bb = bufs[ex["BIAS"]].astype(np.float64)
            if bb.ndim == 1:
                bb = bb.reshape([-1 if i == ch else 1 for i in range(dst.ndim)])
            ref = ref + torch.from_numpy(bb)
        for po in (attr["post"] if attr else []):
            if po[0] == "eltwise":
                ref = _post_eltwise(po[1], ref, po[2], po[3])
            else:
                r = bufs[po[2]].astype(np.float64)
                if len(po) > 3 and po[3] > 0 and bufs[po[2]].dtype.kind in "iu":  # quantized src1
                    r = (r - po[4]) * po[3]
                ref = ref * torch.from_numpy(r) if po[1] == "mul" else ref + torch.from_numpy(r)
        checked += 1
        got = dst
        want = ref.numpy()
        if attr and attr.get("dst_scale"):      # folded requant: integer dst, compare codes
            lo, hi = (-128, 127) if got.dtype == np.int8 else (0, 255)
            want = np.clip(np.rint(want / attr["dst_scale"]) + attr.get("dst_zp", 0), lo, hi)
            # a value on a rounding tie may land on either code (libm vs torch ulps)
            n = int((np.abs(got.astype(np.float64) - want) > 1).sum())
            if n:
                bad.append((op.name, n))
            continue
        if got.dtype == np.int8:
            want = np.clip(want, -128, 127)
        if attr:      # fused f32 epilogue: libm vs torch transcendental ulps
            n = int((np.abs(got - want) > 1e-5 * np.maximum(np.abs(want), 1.0)).sum())
        else:
            n = int((got.astype(np.float64) != want).sum())
        if n:
            bad.append((op.name, n))
    return checked, bad


# ── one call: lower, run, compare ─────────────────────────────────────────

#: pass thresholds (relative L2 vs the mode's reference, plus the same top-1); int8 allows
#: quantization-tie flips, a lowering bug is O(1).
TOL = {"f32": 1e-4, "int8": 0.15}

@dataclass
class ModelReport:
    name: str
    mode: str
    ops: int
    relerr: float
    top1: int
    ref_top1: int
    float_top1: int
    float_relerr: float
    on_aie: int
    impls: dict
    graph: AGraph = None
    gemm_checked: int = 0
    gemm_bad: list = field(default_factory=list)
    batch: int = 1
    top1_miss: int = 0  # images whose top-1 differs from the reference's (top1 / ref_top1: image 0's)

    @property
    def ok(self) -> bool:
        return self.relerr < TOL[self.mode] and self.top1_miss == 0 and not self.gemm_bad

    def __repr__(self) -> str:
        return (f"{self.name:<20} {self.mode:<4} {self.ops:4d} ops  rel {self.relerr:.2e}  top1 {self.top1} "
                f"(ref {self.ref_top1}, float model {self.float_top1}, rel vs float {self.float_relerr:.2e})  "
                f"AIE {self.on_aie} prim  int GEMM exact {self.gemm_checked - len(self.gemm_bad)}/{self.gemm_checked}  "
                f"{'PASS' if self.ok else 'FAIL'}")


def references(model, x, mode, name="model", mlir: bool = False, keep_ir=None):
    """(graph, reference output for the mode, float model output)."""
    import torch
    gm = prepare(model, x)
    with torch.no_grad():
        yf = gm(x)
    yf = (yf[0] if isinstance(yf, (list, tuple)) else yf).numpy().astype(np.float32)
    qscale = calibrate(gm, x) if mode == "int8" else None
    g = lower(model, x, mode, name, gm=gm, qscale=qscale, fuse=not mlir)
    if mlir:
        from .dnn_mlir import optimize
        g = optimize(g, keep_ir=keep_ir)       # keep_ir: <dir>/<name>.in.mlir / .opt.mlir
    qres = False
    if mlir:
        from .dnn_mlir import PIPELINE
        qres = "--aiednn-quantize-residual" in PIPELINE
    ref = yf if mode == "f32" else fake_quant_reference(gm, x, qscale, quant_residual=qres).numpy().astype(np.float32)
    g.top1_range = getattr(model, "top1_range", None)   # e.g. a detector's class-score rows, not its boxes
    return g, ref, yf


def model_test(model, x, mode: str, lib, name: str = "model", mlir: bool = False) -> ModelReport:
    """lower *model*, run it through aiednn on x86 (ctypes), compare with PyTorch."""
    g, ref, yf = references(model, x, mode, name, mlir=mlir)
    y, impls, bufs = run_ctypes(g, lib, x.numpy(), keep=True)
    if g.plan is not None:
        # the MLIR memory plan overwrote earlier inputs: re-run on private buffers for the
        # per-layer GEMM check and require bit-identical outputs
        plan, g.plan = g.plan, None
        y2, _, bufs = run_ctypes(g, lib, x.numpy(), keep=True)
        g.plan = plan
        if not np.array_equal(np.asarray(y).view(np.uint8), np.asarray(y2).view(np.uint8)):
            raise AiegraphError("dnn_torch", f"{name}: the memory plan changes the output (arena vs private buffers)")
    checked, bad = gemm_exact_check(g, bufs)
    rel = float(np.linalg.norm(y - ref) / max(np.linalg.norm(ref), 1e-30))
    frel = float(np.linalg.norm(y - yf) / max(np.linalg.norm(yf), 1e-30))
    counts = {}
    for i in impls:
        counts[i] = counts.get(i, 0) + 1
    yt, rt, ft = _top1(y, g.top1_range), _top1(ref, g.top1_range), _top1(yf, g.top1_range)
    return ModelReport(name, mode, len(g.ops), rel, yt[0], rt[0], ft[0], frel,
                       sum(n for i, n in counts.items() if "aie:" in i), counts, g, checked, bad,
                       batch=len(yt), top1_miss=sum(a != b for a, b in zip(yt, rt)))


def _top1(y, rng=None) -> list:
    """top-1 class per image: one per row of a [batch, classes] output, else over the whole output"""
    y = np.asarray(y)
    rows = y.reshape(y.shape[0], -1) if y.ndim >= 2 and y.shape[0] > 1 else y.reshape(1, -1)
    lo, hi, grp = rng or (0, rows.shape[1], 1)
    return [int(r[lo:hi].argmax()) // grp for r in rows]
