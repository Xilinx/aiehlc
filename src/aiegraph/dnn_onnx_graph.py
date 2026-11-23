###############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
"""ONNX model -> aiednn graph API (aiegraph.dnn_graph): one graph op per ONNX node, initializers as
constant tensors, anything without a mapping as a wildcard op. What an ONNX Runtime execution
provider's GetCapability / Compile do, without onnxruntime: unsupported partitions run through
onnx's ReferenceEvaluator.

    m = OnnxGraph.load(lib, "resnet18-v1-7.int8.qdq.onnx", batch=1)
    m.report()                          # partitions: fused, int8, implementation, unsupported
    y = m.run({"data": x})[m.outputs[0]]
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np

from .dnn_ctypes import conv_params, pool_params
from .dnn_graph import Graph

_ONNX_DT = {1: "f32", 2: "u8", 3: "s8", 6: "s32", 9: "u8", 10: "f16", 16: "bf16"}   # bool: 0 / 1 bytes
_ELTWISE = {"Relu": "relu", "Sigmoid": "sigmoid", "Tanh": "tanh", "HardSwish": "hardswish"}
# unary ONNX ops -> (aiednn eltwise algorithm, alpha, beta); attributes read in _map_ext
_ELT_ALG = {"Abs": "eltwise_abs", "Exp": "eltwise_exp", "Log": "eltwise_log", "Sqrt": "eltwise_sqrt",
            "Neg": "eltwise_neg", "Reciprocal": "eltwise_reciprocal", "Erf": "eltwise_erf", "Floor": "eltwise_floor",
            "Ceil": "eltwise_ceil", "Round": "eltwise_round", "Sign": "eltwise_sign", "Not": "eltwise_logical_not",
            "Sin": "eltwise_sin", "Cos": "eltwise_cos", "Tan": "eltwise_tan", "Softsign": "eltwise_softsign",
            "Mish": "eltwise_mish", "Softplus": "eltwise_soft_relu"}
_BIN_ALG = {"Sub": "binary_sub", "Div": "binary_div", "Max": "binary_max", "Min": "binary_min", "Pow": "binary_pow",
            "Equal": "binary_eq", "Greater": "binary_gt", "GreaterOrEqual": "binary_ge", "Less": "binary_lt",
            "LessOrEqual": "binary_le", "And": "binary_logical_and", "Or": "binary_logical_or",
            "Xor": "binary_logical_xor"}
_REDUCE = {"ReduceSum": ("reduction_sum", 0.0), "ReduceMean": ("reduction_mean", 0.0),
           "ReduceMax": ("reduction_max", 0.0), "ReduceMin": ("reduction_min", 0.0),
           "ReduceProd": ("reduction_mul", 0.0), "ReduceL1": ("reduction_norm_lp_power_p_sum", 1.0),
           "ReduceL2": ("reduction_norm_lp_sum", 2.0), "ReduceSumSquare": ("reduction_norm_lp_power_p_sum", 2.0)}


def _attrs(node) -> dict:
    import onnx
    return {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}


class OnnxGraph:
    def __init__(self, lib, model, batch: int = 1, layouts: bool = True):
        import onnx
        from onnx import numpy_helper, shape_inference
        for i in list(model.graph.input) + list(model.graph.output):   # pin a symbolic / 0 batch
            d = i.type.tensor_type.shape.dim
            if d and not d[0].dim_value:
                d[0].dim_value = batch
        del model.graph.value_info[:]   # stale annotations (batch 0, empty shapes) block inference
        model.ir_version = max(model.ir_version, 4)   # IR < 4: initializers count only when listed as inputs
        self.model = shape_inference.infer_shapes(model, strict_mode=True)
        g = self.model.graph
        self.consts = {t.name: numpy_helper.to_array(t) for t in g.initializer}
        for n in g.node:
            if n.op_type == "Constant":
                self.consts[n.output[0]] = numpy_helper.to_array(_attrs(n)["value"])
        self.info = {}
        for v in list(g.input) + list(g.value_info) + list(g.output):
            tt = v.type.tensor_type
            self.info[v.name] = (tuple(d.dim_value for d in tt.shape.dim), _ONNX_DT.get(tt.elem_type))
        for k, a in self.consts.items():
            self.info[k] = (tuple(a.shape), _ONNX_DT.get(onnx.helper.np_dtype_to_tensor_dtype(a.dtype)))
        self.inputs = [i.name for i in g.input if i.name not in self.consts]
        self.outputs = [o.name for o in g.output]
        self.graph = Graph(lib, layouts=layouts)
        self.nodes = {}
        self.wildcards = Counter()
        for k, n in enumerate(g.node):
            if n.op_type != "Constant":
                self._add(k, n)
        for o in self.outputs:
            self.graph.mark_output(o)
        self.parts = self.graph.compile_all()

    @classmethod
    def load(cls, lib, path, batch: int = 1, layouts: bool = True) -> "OnnxGraph":
        import onnx
        return cls(lib, onnx.load(str(path)), batch, layouts)

    # ── mapping ──────────────────────────────────────────────────────────
    def _t(self, name) -> str:
        if name not in self.graph.tensors:
            shape, dt = self.info[name]
            self.graph.tensor(name, shape, dt or "f32", name in self.consts)
        return name

    def _scalar(self, name, default=0):
        return default if not name else self.consts[name].reshape(-1)

    def _add(self, k, n):
        ins = [i for i in n.input if i]
        a = _attrs(n)
        name = n.name or f"{n.op_type}_{k}"
        self.nodes[name] = n
        out = n.output[0]
        op, kw, gins = self._map(n, ins, a)
        # only what the graph op reads must have a graph type (int64 axes / starts are read here, as constants)
        chk = (gins if op is not None else ins) + [out]
        if op is None or len(n.output) != 1 or any(i not in self.info for i in chk) or \
                any(self.info[t][1] is None for t in chk):
            op, kw, gins = "wildcard", {}, ins
            self.wildcards[n.op_type] += 1
        for t in gins + [out]:
            self._t(t)
        self.graph.add(op, gins, out, name=name, **kw)

    def _map(self, n, ins, a):
        t = n.op_type
        if t in ("QuantizeLinear", "DequantizeLinear"):
            kind = "quantize" if t == "QuantizeLinear" else "dequantize"
            if ins[1] not in self.consts or (len(ins) > 2 and ins[2] not in self.consts):   # computed in the graph
                return "dynamic_" + kind, dict(axis=int(a.get("axis", 1)) if self.info[ins[1]][0] not in ((), (1,))
                                               else -1), ins
            s = self._scalar(ins[1], None)
            z = self._scalar(ins[2] if len(ins) > 2 else None, np.zeros(1, np.int32))
            if s.size == 1:
                return kind, dict(scale=float(s[0]), zero_point=int(z[0])), ins[:1]
            if kind == "quantize":
                return None, {}, ins
            return kind, dict(scales=s.astype(np.float32), zero_points=np.broadcast_to(z, s.shape).astype(np.int32),
                              axis=int(a.get("axis", 1))), ins[:1]
        if t == "Conv":
            ks = a.get("kernel_shape") or self.info[ins[1]][0][2:]
            if len(ks) != 2:
                return None, {}, ins
            p = a.get("pads", [0, 0, 0, 0])
            d = a.get("dilations", [1, 1])
            cp = conv_params(tuple(a.get("strides", [1, 1])), (p[0], p[1]), (p[2], p[3]), (d[0] - 1, d[1] - 1),
                             a.get("group", 1))
            return "convolution", dict(conv=cp), ins
        if t == "Gemm":
            if a.get("alpha", 1.0) != 1.0 or a.get("beta", 1.0) != 1.0 or a.get("transA", 0):
                return None, {}, ins
            return "matmul", dict(weights_transposed=int(a.get("transB", 0))), ins
        if t == "MatMul":
            return "matmul", {}, ins
        if t in _ELTWISE:
            return _ELTWISE[t], {}, ins
        if t == "Clip" and len(ins) == 3 and all(i in self.consts for i in ins[1:]):
            return "clamp", dict(alpha=float(self.consts[ins[1]]), beta=float(self.consts[ins[2]])), ins[:1]
        if t in ("Add", "Mul"):
            return ("add" if t == "Add" else "multiply"), {}, ins
        if t in ("MaxPool", "AveragePool"):
            p = a.get("pads", [0, 0, 0, 0])
            if a.get("ceil_mode", 0):
                return None, {}, ins
            pp = pool_params(tuple(a["kernel_shape"]), tuple(a.get("strides", [1, 1])), (p[0], p[1]), (p[2], p[3]))
            kw = dict(pool=pp)
            if t == "AveragePool" and not a.get("count_include_pad", 0):
                kw["alg"] = "pooling_avg_exclude_padding"
            return ("max_pool" if t == "MaxPool" else "avg_pool"), kw, ins
        if t == "GlobalAveragePool":
            h, w = self.info[ins[0]][0][2:]
            return "avg_pool", dict(pool=pool_params((h, w), (1, 1))), ins
        if t in ("Flatten", "Reshape", "Squeeze", "Unsqueeze"):
            return "reshape", {}, ins[:1]
        if t == "Transpose":
            return "transpose", dict(perm=tuple(a["perm"])), ins
        if t == "Softmax":
            return "softmax", dict(axis=int(a.get("axis", -1))), ins
        if t == "Concat":
            return "concat", dict(axis=int(a["axis"])), ins
        return self._map_ext(n, ins, a)

    def _map_ext(self, n, ins, a):
        """the rest of the standard ONNX operator set the graph API serves"""
        t, r = n.op_type, len(self.info[ins[0]][0]) if ins and ins[0] in self.info else 0
        if t in _ELT_ALG:
            return "eltwise", dict(alg=_ELT_ALG[t], alpha=1.0), ins
        if t in ("Elu", "LeakyRelu"):
            return "eltwise", dict(alg="eltwise_elu" if t == "Elu" else "eltwise_relu",
                                   alpha=float(a.get("alpha", 1.0 if t == "Elu" else 0.01))), ins
        if t == "HardSigmoid":
            return "eltwise", dict(alg="eltwise_hardsigmoid", alpha=float(a.get("alpha", 0.2)),
                                   beta=float(a.get("beta", 0.5))), ins
        if t == "Gelu":
            return "gelu", dict(alg="eltwise_gelu_tanh" if a.get("approximate", b"none") in (b"tanh", "tanh") else 0), ins
        if t in _BIN_ALG and len(ins) == 2:
            return "binary", dict(alg=_BIN_ALG[t]), ins
        if t == "Mod" and a.get("fmod", 0) and len(ins) == 2:
            return "binary", dict(alg="binary_fmod"), ins
        if t == "Where":
            return "select", {}, ins
        if t == "Cast":
            return "type_cast", {}, ins
        if t == "Identity":
            return "reshape", {}, ins
        if t == "Dropout" and len(n.output) == 1:   # inference: the identity
            return "dropout", dict(alpha=0.0), ins[:1]
        if t == "LogSoftmax":
            return "softmax", dict(axis=int(a.get("axis", -1)), alg="softmax_log"), ins
        if t in _REDUCE:
            axes = a.get("axes")
            if axes is None and len(ins) > 1:
                if ins[1] not in self.consts:
                    return None, {}, ins
                axes = self.consts[ins[1]].reshape(-1).tolist()
            if axes is None:
                axes = [] if a.get("noop_with_empty_axes", 0) else list(range(r))
            alg, p = _REDUCE[t]
            return "reduction", dict(alg=alg, pnorm=p, reduce_mask=sum(1 << (int(x) % r) for x in axes)), ins[:1]
        if t == "LayerNormalization" and int(a.get("axis", -1)) in (-1, r - 1):
            return "layer_norm", dict(eps=float(a.get("epsilon", 1e-5))), ins
        if t == "BatchNormalization" and len(ins) == 5:   # x, scale, B, mean, var -> x, mean, var, scale, shift
            return "batch_norm_inference", dict(eps=float(a.get("epsilon", 1e-5))), [ins[0], ins[3], ins[4], ins[1], ins[2]]
        if t == "InstanceNormalization":
            return "group_norm", dict(groups=int(self.info[ins[0]][0][1]), eps=float(a.get("epsilon", 1e-5))), ins
        if t == "PRelu":
            return "prelu", {}, ins
        if t == "Slice":
            return self._slice(ins, r)
        if t == "ConvTranspose":
            return self._conv_transpose(ins, a)
        return None, {}, ins

    def _slice(self, ins, r):
        """constant starts / ends / axes, step 1 -> slice (starts per axis, the output's dims)"""
        if len(ins) < 3 or any(i not in self.consts for i in ins[1:]):
            return None, {}, ins
        starts = self.consts[ins[1]].reshape(-1)
        axes = self.consts[ins[3]].reshape(-1) if len(ins) > 3 else np.arange(len(starts))
        if len(ins) > 4 and np.any(self.consts[ins[4]] != 1):
            return None, {}, ins
        st = [0] * r
        for ax, s0 in zip(axes, starts):
            d = self.info[ins[0]][0][int(ax) % r]
            st[int(ax) % r] = int(min(max(s0 + d if s0 < 0 else s0, 0), d))
        return "slice", dict(starts=st), ins[:1]

    def _conv_transpose(self, ins, a):
        """ONNX weights [C, O/g, KH, KW] -> aiednn deconvolution [O, C/g, KH, KW] (group 1, a constant)"""
        w = self.consts.get(ins[1])
        if w is None or w.ndim != 4 or a.get("group", 1) != 1 or any(a.get("output_padding", [0, 0])) or \
                a.get("output_shape") or any(d != 1 for d in a.get("dilations", [1, 1])):
            return None, {}, ins
        name = ins[1] + "__oihw"
        if name not in self.consts:
            self.consts[name] = np.ascontiguousarray(w.transpose(1, 0, 2, 3))
            self.info[name] = (self.consts[name].shape, "f32")
        p = a.get("pads", [0, 0, 0, 0])
        cp = conv_params(tuple(a.get("strides", [1, 1])), (p[0], p[1]), (p[2], p[3]))
        return "deconvolution", dict(conv=cp), [ins[0], name] + ins[2:]

    # ── running ──────────────────────────────────────────────────────────
    def _fallback(self, part, env):
        """an unsupported partition through onnx's reference evaluator"""
        from onnx import helper
        from onnx.reference import ReferenceEvaluator
        nodes = [self.nodes[o] for o in part.ops]
        feeds = {i: env[i] for n in nodes for i in n.input if i and i in env}
        ins = [helper.make_tensor_value_info(k, helper.np_dtype_to_tensor_dtype(v.dtype), v.shape)
               for k, v in feeds.items()]
        outs = [helper.make_tensor_value_info(o, 0, None) for o in part.outputs]
        sub = helper.make_model(helper.make_graph(nodes, "part", ins, outs), opset_imports=self.model.opset_import)
        res = ReferenceEvaluator(sub).run(None, feeds)
        for o, r in zip(part.outputs, res):
            env[o][...] = r

    def run(self, feeds: dict) -> dict:
        env = dict(self.consts)
        env.update(feeds)
        return self.graph.run(env, self._fallback)

    def report(self) -> str:
        sup = [p for p in self.parts if p.supported]
        lines = [f"{len(self.parts)} partitions: {len(sup)} supported, {len(self.parts) - len(sup)} unsupported; "
                 f"{sum(len(p.ops) for p in sup)} / {sum(len(p.ops) for p in self.parts)} ONNX nodes inside "
                 f"supported partitions"]
        impls = Counter((p.info.split("-> ")[-1] if p.supported else "UNSUPPORTED") for p in self.parts)
        lines += [f"  {n:4d}  {k}" for k, n in impls.most_common()]
        if self.wildcards:
            lines.append("  no mapping: " + ", ".join(f"{k} x{n}" for k, n in self.wildcards.most_common()))
        for p in self.parts:
            lines.append(f"  {'+' if p.supported else '-'} {p.info}")
        return "\n".join(lines)
