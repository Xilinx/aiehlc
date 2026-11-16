###############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
"""ctypes binding of include/aiednn_graph.h (the partition graph API)."""

from __future__ import annotations

import ctypes as C
import subprocess
from pathlib import Path

import numpy as np

from .dnn_ctypes import DT, MD, REPO, AiednnLib, ConvParams, PoolParams, md

GOP = {k: i + 1 for i, k in enumerate(
    ["convolution", "matmul", "inner_product", "bias_add", "add", "multiply", "binary", "relu", "clamp", "gelu",
     "sigmoid", "tanh", "hardswish", "eltwise", "max_pool", "avg_pool", "softmax", "layer_norm",
     "batch_norm_inference", "concat", "reshape", "transpose", "quantize", "dequantize", "sdpa", "wildcard",
     "deconvolution", "resampling", "reduction", "group_norm", "prelu", "slice",
     "type_cast", "reorder", "select", "squared_difference", "dropout", "rms_norm",
     "eltwise_backward", "softmax_backward", "pool_backward", "resampling_backward", "bias_add_backward",
     "convolution_backward_data", "convolution_backward_weights", "deconvolution_backward_data",
     "deconvolution_backward_weights", "batch_norm_training", "batch_norm_backward", "layer_norm_backward",
     "group_norm_backward", "prelu_backward", "gen_index", "dynamic_quantize", "dynamic_dequantize",
     "linear_attention"])}


class LT(C.Structure):
    _fields_ = [("id", C.c_uint64), ("md", MD), ("property", C.c_int)]


class GAttr(C.Structure):
    _fields_ = [("alg", C.c_int), ("alpha", C.c_float), ("beta", C.c_float), ("conv", ConvParams),
                ("pool", PoolParams), ("weights_transposed", C.c_int), ("axis", C.c_int), ("perm", C.c_int * 6),
                ("eps", C.c_float), ("flags", C.c_uint), ("scale", C.c_float), ("zero_point", C.c_int32),
                ("n_scales", C.c_int64), ("scales", C.POINTER(C.c_float)), ("zero_points", C.POINTER(C.c_int32)),
                ("sdpa_scale", C.c_float), ("causal", C.c_int),
                ("groups", C.c_int64), ("pnorm", C.c_float), ("reduce_mask", C.c_uint),
                ("starts", C.c_int64 * 6), ("seed", C.c_uint64)]


class GTensor(C.Structure):
    _fields_ = [("id", C.c_uint64), ("data", C.c_void_p)]


class ImplPerf(C.Structure):
    _fields_ = [("impl", C.c_char * 64), ("on_aie", C.c_int), ("time_us", C.c_float), ("workspace_size", C.c_size_t),
                ("status", C.c_int)]


FIND_HEURISTIC, FIND_MEASURE = 0, 1


def build_graph_lib(out: Path, mesh=None) -> AiednnLib:
    """AiednnLib.build plus the graph API"""
    from .dnn import impl_list_source
    from .kernel_lib import kernel_lib
    out.mkdir(parents=True, exist_ok=True)
    lib = kernel_lib(REPO / "src" / "aietensorop", quiet=True)
    picked = [e for e in lib if e.contract.source.name == "gemm_uk.cc" and "vek280" in e.contract.spec.targets]
    (out / "impl_list.cc").write_text(impl_list_source(picked, x86=True, mesh=mesh, count=bool(picked)))
    (out / "graph_tu.cc").write_text('#include "aiednn_rt.cc"\n#include "aiednn_rt_graph.h"\n')
    so = out / "libaiednn_graph.so"
    mdefs = [f"-DHW_ROWS={mesh[0]}", f"-DHW_COLS={mesh[1]}"] if mesh else []
    cmd = ["g++", "-O2", "-shared", "-fPIC", "-std=gnu++17", *mdefs, f"-I{REPO / 'include'}",
           f"-I{REPO / 'src' / 'aietensorop' / 'ops'}", f"-I{REPO / 'src' / 'aiednn'}", str(out / "graph_tu.cc"),
           str(out / "impl_list.cc"), "-x", "c", str(REPO / "src" / "aietensorop" / "gemm" / "gemm_uk_ref.c"),
           "-x", "none", "-o", str(so), "-lm"]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode:
        raise RuntimeError(res.stderr)
    lib = AiednnLib(so)
    s = lib.so
    for fn in ("aiednn_partition_info", "aiednn_compiled_partition_impl_info"):
        getattr(s, fn).restype = C.c_char_p
    s.aiednn_compiled_partition_get_workspace_size.restype = C.c_size_t
    s.aiednn_compiled_partition_execute.argtypes = [C.c_void_p, C.c_void_p, C.c_int, C.c_void_p, C.c_int, C.c_void_p,
                                                    C.c_void_p, C.c_size_t]
    return lib


def aligned_workspace(nbytes: int, align: int = 64) -> np.ndarray:
    """a uint8 view of nbytes starting on an align-byte boundary (what execute_async requires)"""
    raw = np.empty(max(nbytes, 1) + align, np.uint8)
    off = (-raw.ctypes.data) % align
    return raw[off:off + max(nbytes, 1)]


class Stream:
    def __init__(self, lib: AiednnLib):
        self.lib, self.h = lib, C.c_void_p()
        st = lib.so.aiednn_stream_create(C.byref(self.h), 0)
        if st:
            raise RuntimeError(lib.status(st))

    def wait(self) -> int:
        return self.lib.so.aiednn_stream_wait(self.h)

    def __del__(self):
        try:
            self.lib.so.aiednn_stream_destroy(self.h)
        except Exception:
            pass


def attr(lib: AiednnLib, **kw) -> GAttr:
    """GAttr from keywords: alg"""
    a = GAttr()
    lib.so.aiednn_graph_op_attr_init(C.byref(a))
    keep = []
    for k, v in kw.items():
        if k == "alg":
            a.alg = lib.ALG[v] if isinstance(v, str) else int(v)
        elif k == "starts":
            for i, v in enumerate(v):
                a.starts[i] = int(v)
        elif k == "perm":
            for i, p in enumerate(v):
                a.perm[i] = int(p)
        elif k == "scales":
            arr = np.ascontiguousarray(v, np.float32)
            keep.append(arr)
            a.n_scales, a.scales = arr.size, arr.ctypes.data_as(C.POINTER(C.c_float))
        elif k == "zero_points":
            arr = np.ascontiguousarray(v, np.int32)
            keep.append(arr)
            a.zero_points = arr.ctypes.data_as(C.POINTER(C.c_int32))
        else:
            setattr(a, k, v)
    a._keep = keep  # the arrays live until add_op copies them
    return a


class Partition:
    def __init__(self, graph: "Graph", handle):
        self.graph, self.h = graph, handle
        so = graph.lib.so
        self.supported = bool(so.aiednn_partition_is_supported(handle))
        self.info = so.aiednn_partition_info(handle).decode()
        n = so.aiednn_partition_get_ops(handle, None, 0)
        ids = (C.c_uint64 * max(n, 1))()
        so.aiednn_partition_get_ops(handle, ids, n)
        self.ops = [graph.op_name[i] for i in ids[:n]]
        self.inputs = self._lts(so.aiednn_partition_get_inputs)
        self.outputs = self._lts(so.aiednn_partition_get_outputs)

    def _lts(self, fn) -> list:
        n = fn(self.h, None, 0)
        arr = (LT * max(n, 1))()
        fn(self.h, arr, n)
        return [self.graph.tensor_name[arr[i].id] for i in range(n)]

    def compile(self) -> "Compiled":
        cp = C.c_void_p()
        st = self.graph.lib.so.aiednn_partition_compile(self.h, C.byref(cp))
        if st:
            raise RuntimeError(f"compile {self.info}: {self.graph.lib.status(st)}")
        return Compiled(self.graph, cp, self)

    def __repr__(self):
        return f"Partition({'+' if self.supported else '-'} {self.info})"


class Compiled:
    def __init__(self, graph: "Graph", handle, part: Partition):
        self.graph, self.h, self.part = graph, handle, part
        self.impl = graph.lib.so.aiednn_compiled_partition_impl_info(handle).decode()

    @property
    def workspace_size(self) -> int:
        return self.graph.lib.so.aiednn_compiled_partition_get_workspace_size(self.h)

    def find(self, mode: int = FIND_HEURISTIC) -> list:
        """[(impl, on_aie, time_us, workspace_size, status)], fastest first under FIND_MEASURE"""
        so = self.graph.lib.so
        n = so.aiednn_compiled_partition_find(self.h, mode, None, 0) if mode == FIND_HEURISTIC else 32
        arr = (ImplPerf * max(n, 1))()
        n = min(so.aiednn_compiled_partition_find(self.h, mode, arr, len(arr)), len(arr))
        self.impl = so.aiednn_compiled_partition_impl_info(self.h).decode()
        return [(r.impl.decode(), bool(r.on_aie), r.time_us, r.workspace_size, r.status) for r in arr[:n]]

    def set_impl(self, name: str) -> None:
        st = self.graph.lib.so.aiednn_compiled_partition_set_impl(self.h, name.encode())
        if st:
            raise RuntimeError(f"set_impl {name}: {self.graph.lib.status(st)}")
        self.impl = self.graph.lib.so.aiednn_compiled_partition_impl_info(self.h).decode()

    def execute(self, ins: dict, outs: dict, stream=None, workspace: np.ndarray = None) -> None:
        """workspace: a uint8 array of >= workspace_size bytes, 64-byte aligned (see aligned_workspace)"""
        tid = self.graph.tensor_id
        def arr(d):
            a = (GTensor * max(len(d), 1))()
            for i, (k, v) in enumerate(d.items()):
                a[i].id, a[i].data = tid[k], v.ctypes.data
            return a
        ws, wn = (workspace.ctypes.data, workspace.nbytes) if workspace is not None else (None, 0)
        st = self.graph.lib.so.aiednn_compiled_partition_execute(self.h, stream, len(ins), arr(ins), len(outs),
                                                                 arr(outs), C.c_void_p(ws), C.c_size_t(wn))
        if st:
            raise RuntimeError(f"execute {self.part.info}: {self.graph.lib.status(st)}")

    def __del__(self):
        try:
            self.graph.lib.so.aiednn_compiled_partition_destroy(self.h)
        except Exception:
            pass


class Graph:
    def __init__(self, lib: AiednnLib, fuse: bool = True, layouts: bool = True):
        self.lib = lib
        self.h = C.c_void_p()
        st = lib.so.aiednn_graph_create(C.byref(self.h))
        if st:
            raise RuntimeError(lib.status(st))
        self.fuse = int(fuse)    # 0 none, 1 exact (default), 2 also inexact (aiednn_graph_set_fusion)
        lib.so.aiednn_graph_set_fusion(self.h, self.fuse)
        self.layouts = bool(layouts)    # nhwc on internal edges (aiednn_graph_set_layout_propagation)
        lib.so.aiednn_graph_set_layout_propagation(self.h, int(self.layouts))
        self.tensors = {}       # name -> (shape, dt, const)
        self.tensor_id, self.tensor_name, self.op_name = {}, {}, {}
        self.pending = []       # ops before every tensor is described

    def tensor(self, name, shape, dt: str, const: bool = False) -> None:
        self.tensors[name] = (tuple(shape), dt, const)
        if name not in self.tensor_id:
            i = len(self.tensor_id) + 1
            self.tensor_id[name], self.tensor_name[i] = i, name

    def add(self, kind: str, ins: list, out, name=None, **kw) -> None:
        self.pending.append((kind, list(ins), out, name if name is not None else f"{kind}{len(self.pending)}", kw))

    def mark_output(self, name) -> None:
        self.pending.append(("__mark__", [], name, None, {}))

    def _lt(self, name) -> LT:
        shape, dt, const = self.tensors[name]
        lt = LT()
        lt.id, lt.md, lt.property = self.tensor_id[name], md(shape, dt), int(const)
        return lt

    def finalize(self) -> list:
        so = self.lib.so
        marks = []
        for k, (kind, ins, out, name, kw) in enumerate(self.pending):
            if kind == "__mark__":
                marks.append(out)
                continue
            lin = (LT * max(len(ins), 1))(*[self._lt(t) for t in ins])
            outs = list(out) if isinstance(out, (list, tuple)) else [out]   # training ops: several
            lout = (LT * len(outs))(*[self._lt(t) for t in outs])
            a = attr(self.lib, **kw)
            st = so.aiednn_graph_add_op(self.h, C.c_uint64(k + 1), GOP[kind], C.byref(a), len(ins), lin, len(outs),
                                        lout)
            if st:
                raise RuntimeError(f"add_op {name} ({kind}): {self.lib.status(st)}")
            self.op_name[k + 1] = name
        for m in marks:
            so.aiednn_graph_mark_output(self.h, C.c_uint64(self.tensor_id[m]))
        st = so.aiednn_graph_finalize(self.h)
        if st:
            raise RuntimeError(f"finalize: {self.lib.status(st)}")
        n = so.aiednn_graph_get_partitions(self.h, None, 0)
        arr = (C.c_void_p * max(n, 1))()
        so.aiednn_graph_get_partitions(self.h, arr, n)
        return [Partition(self, C.c_void_p(arr[i])) for i in range(n)]

    def run(self, feeds: dict, fallback=None) -> dict:
        """Every partition in order on numpy buffers."""
        env = dict(feeds)
        for p in self.parts:
            for o in p.outputs:
                if o not in env:
                    shape, dt, _ = self.tensors[o]
                    env[o] = np.zeros(shape, DT[dt][1])
            if not p.supported:
                if fallback is None:
                    raise RuntimeError(f"unsupported partition: {p.info}")
                fallback(p, env)
                continue
            if p not in self.compiled:
                self.compiled[p] = p.compile()
            self.compiled[p].execute({k: env[k] for k in p.inputs}, {k: env[k] for k in p.outputs})
        return env

    def compile_all(self) -> list:
        self.parts = self.finalize()
        self.compiled = {}
        return self.parts

    def __del__(self):
        try:
            self.lib.so.aiednn_graph_destroy(self.h)
        except Exception:
            pass
