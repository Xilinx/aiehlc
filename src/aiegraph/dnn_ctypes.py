###############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
"""ctypes binding of include/aiednn.h"""

from __future__ import annotations

import ctypes as C
import re
import subprocess
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]

DT = {"s8": (1, np.int8), "u8": (2, np.uint8), "s32": (3, np.int32), "f32": (4, np.float32),
      # narrow floats: numpy has no bf16 / f8, so their buffers are the raw bits (torch converts)
      "bf16": (5, np.uint16), "f16": (6, np.float16), "f8_e5m2": (7, np.uint8), "f8_e4m3": (8, np.uint8),
      "s4": (9, np.uint8), "u4": (10, np.uint8),   # 4-bit: two per byte (pack4 / unpack4)
      "e8m0": (11, np.uint8), "f4_e2m1": (12, np.uint8)}
ARG = {"src": 1, "src_1": 2, "src_2": 3, "weights": 4, "bias": 5, "dst": 6, "mean": 7, "variance": 8,
       "scale": 9, "shift": 10, "attn_mask": 11, "rope_cos": 48, "rope_sin": 49}
MULTIPLE_SRC = 64


class MD(C.Structure):
    _fields_ = [("ndims", C.c_int), ("dims", C.c_int64 * 6), ("data_type", C.c_int), ("format", C.c_int)]


FMT = {"dense": 0, "nhwc": 1}   # aiednn_format_t


class ExecArg(C.Structure):
    _fields_ = [("arg", C.c_int), ("data", C.c_void_p)]


class PostOp(C.Structure):
    _fields_ = [("kind", C.c_int), ("alg", C.c_int), ("alpha", C.c_float), ("beta", C.c_float), ("src1", MD),
                ("q_scale", C.c_float), ("q_zero_point", C.c_int32), ("q_data_type", C.c_int)]


class Attr(C.Structure):
    _fields_ = [("n_scales", C.c_int64), ("scales", C.POINTER(C.c_float)), ("dst_scale", C.c_float),
                ("dst_zero_point", C.c_int32), ("n_post_ops", C.c_int), ("post_ops", PostOp * 4),
                ("src_zero_point", C.c_int32), ("wei_zero_point", C.c_int32),
                ("wei_scales", MD), ("wei_zero_points", MD),
                ("src_scales", C.c_float * 2), ("src_zero_points", C.c_int32 * 2),
                ("src_scales_23", C.c_float * 2), ("src_zero_points_23", C.c_int32 * 2), ("flags", C.c_uint)]


POST_OP_SRC = 128
ARG_WEI_SCALES, ARG_WEI_ZPS = 136, 137


def attr(lib_alg: dict, scales=None, dst_scale=0.0, dst_zp=0, post_ops=(), src_zp=0, wei_zp=0,
         wei_scales=None, wei_zps=None, flags=0):
    """post_ops: ("eltwise", alg, alpha, beta) | ("binary", alg, shape, dt[, scale, zp]) |"""
    a = Attr()
    a.src_zero_point, a.wei_zero_point, a.flags = src_zp, wei_zp, flags   # flags: AIEDNN_ATTR_* opt-ins
    if wei_scales is not None:
        a.wei_scales = md(*wei_scales)
    if wei_zps is not None:
        a.wei_zero_points = md(*wei_zps)
    keep = []
    if scales is not None:
        arr = np.ascontiguousarray(scales, np.float32)
        keep.append(arr)
        a.n_scales = arr.size
        a.scales = arr.ctypes.data_as(C.POINTER(C.c_float))
    a.dst_scale, a.dst_zero_point, a.n_post_ops = dst_scale, dst_zp, len(post_ops)
    for k, po in enumerate(post_ops):
        if po[0] == "eltwise":
            a.post_ops[k].kind, a.post_ops[k].alg = 1, lib_alg["eltwise_" + po[1]]
            a.post_ops[k].alpha, a.post_ops[k].beta = po[2], po[3]
        elif po[0] == "requant_eltwise":   # ("requant_eltwise", alg, alpha, beta, q_scale, q_zp, q_dt)
            a.post_ops[k].kind, a.post_ops[k].alg = 4, lib_alg["eltwise_" + po[1]]
            a.post_ops[k].alpha, a.post_ops[k].beta = po[2], po[3]
            a.post_ops[k].q_scale, a.post_ops[k].q_zero_point, a.post_ops[k].q_data_type = po[4], po[5], DT[po[6]][0]
        elif po[0] == "requant_binary":   # ("requant_binary", alg, shape, dt, scale, zp, q_scale, q_zp, q_dt)
            a.post_ops[k].kind, a.post_ops[k].alg = 3, lib_alg["binary_" + po[1]]
            a.post_ops[k].src1 = md(po[2], po[3])
            a.post_ops[k].alpha, a.post_ops[k].beta = po[4], po[5]
            a.post_ops[k].q_scale, a.post_ops[k].q_zero_point, a.post_ops[k].q_data_type = po[6], po[7], DT[po[8]][0]
        else:   # ("binary", alg, shape, dt[, scale, zp[, format]])
            a.post_ops[k].kind, a.post_ops[k].alg = 2, lib_alg["binary_" + po[1]]
            a.post_ops[k].src1 = md(po[2], po[3], po[6] if len(po) > 6 else "dense")
            if len(po) > 4:
                a.post_ops[k].alpha, a.post_ops[k].beta = po[4], po[5]
    return a, keep


class ConvParams(C.Structure):
    _fields_ = [("strides", C.c_int64 * 2), ("padding_l", C.c_int64 * 2), ("padding_r", C.c_int64 * 2),
                ("dilates", C.c_int64 * 2), ("groups", C.c_int64)]


class PoolParams(C.Structure):
    _fields_ = [("kernel", C.c_int64 * 2), ("strides", C.c_int64 * 2), ("padding_l", C.c_int64 * 2),
                ("padding_r", C.c_int64 * 2), ("dilates", C.c_int64 * 2)]


def _alg_table() -> dict:
    """aiednn_alg_kind_t values parsed from the header (one source of truth)."""
    text = (REPO / "include" / "aiednn.h").read_text()
    body = text[text.index("aiednn_alg_undef = 0,"):text.index("} aiednn_alg_kind_t;")]
    out, val = {}, 0
    for m in re.finditer(r"aiednn_(\w+)(?:\s*=\s*(\d+))?,", body):
        val = int(m.group(2)) if m.group(2) else val + 1
        out[m.group(1)] = val
    return out


def md(shape, dt: str, fmt: str = "dense") -> MD:
    m = MD()
    m.ndims = len(shape)
    for i, d in enumerate(shape):
        m.dims[i] = int(d)
    m.data_type = DT[dt][0]
    m.format = FMT[fmt]
    return m


def conv_params(stride=(1, 1), pad_l=(0, 0), pad_r=(0, 0), dil=(0, 0), groups=1) -> ConvParams:
    c = ConvParams()
    for i in range(2):
        c.strides[i], c.padding_l[i], c.padding_r[i], c.dilates[i] = stride[i], pad_l[i], pad_r[i], dil[i]
    c.groups = groups
    return c


def pool_params(kernel, stride=(1, 1), pad_l=(0, 0), pad_r=(0, 0), dil=(0, 0)) -> PoolParams:
    p = PoolParams()
    for i in range(2):
        p.kernel[i], p.strides[i], p.padding_l[i], p.padding_r[i], p.dilates[i] = (
            kernel[i], stride[i], pad_l[i], pad_r[i], dil[i])
    return p


class ConvNdParams(C.Structure):
    _fields_ = [("strides", C.c_int64 * 3), ("padding_l", C.c_int64 * 3), ("padding_r", C.c_int64 * 3),
                ("dilates", C.c_int64 * 3), ("groups", C.c_int64)]


class PoolNdParams(C.Structure):
    _fields_ = [("kernel", C.c_int64 * 3), ("strides", C.c_int64 * 3), ("padding_l", C.c_int64 * 3),
                ("padding_r", C.c_int64 * 3), ("dilates", C.c_int64 * 3)]


def conv_nd_params(stride, pad_l, pad_r, dil, groups=1) -> ConvNdParams:
    """aiednn_conv_nd_params_t: entry i = spatial dim i (len(stride) = 1..3)"""
    c = ConvNdParams()
    for i in range(len(stride)):
        c.strides[i], c.padding_l[i], c.padding_r[i], c.dilates[i] = stride[i], pad_l[i], pad_r[i], dil[i]
    c.groups = groups
    return c


def pool_nd_params(kernel, stride, pad_l, pad_r, dil=None) -> PoolNdParams:
    p = PoolNdParams()
    for i in range(len(kernel)):
        p.kernel[i], p.strides[i], p.padding_l[i], p.padding_r[i] = kernel[i], stride[i], pad_l[i], pad_r[i]
        p.dilates[i] = dil[i] if dil else 0
    return p


class AiednnLib:
    def __init__(self, path: Path):
        self.so = C.CDLL(str(path))
        self.ALG = _alg_table()
        self.so.aiednn_primitive_impl_info.restype = C.c_char_p
        self.so.aiednn_status_str.restype = C.c_char_p

    @classmethod
    def build(cls, out: Path, with_gemm: bool = True, defines=(), mesh=None) -> "AiednnLib":
        """*mesh*: (rows, cols) whose block shapes the impl list carries"""
        from .dnn import impl_list_source
        from .kernel_lib import kernel_lib
        out.mkdir(parents=True, exist_ok=True)
        lib = kernel_lib(REPO / "src" / "aietensorop", quiet=True)
        # every entry of the GEMM megakernel (one source: GEMM + its modes), as a board build picks
        # them, so the x86 dispatch is the board's; counted per entry (entry_calls)
        picked = [e for e in lib if e.contract.source.name == "gemm_uk.cc"
                  and "vek280" in e.contract.spec.targets] if with_gemm else []
        (out / "impl_list.cc").write_text(impl_list_source(picked, x86=True, mesh=mesh, count=bool(picked)))
        so = out / ("libaiednn.so" if with_gemm else "libaiednn_ref.so")
        mdefs = [f"HW_ROWS={mesh[0]}", f"HW_COLS={mesh[1]}"] if mesh else []  # whole-launch stand-ins (gemm_uk_ref.c)
        cmd = ["g++", "-O2", "-shared", "-fPIC", "-std=gnu++17", *[f"-D{d}" for d in (*defines, *mdefs)],
               f"-I{REPO / 'include'}",
               f"-I{REPO / 'src' / 'aietensorop' / 'ops'}",
               f"-I{REPO / 'src' / 'aiednn'}", str(REPO / "src" / "aiednn" / "aiednn_rt.cc"),
               str(out / "impl_list.cc"), "-x", "c", str(REPO / "src" / "aietensorop" / "gemm" / "gemm_uk_ref.c"),
               "-x", "none", "-o", str(so), "-lm"]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode:
            raise RuntimeError(res.stderr)
        return cls(so)

    def entry_calls(self, reset: bool = True) -> dict:
        """{AIE entry: calls since the last reset} (the counting impl list); *reset* after reading"""
        so = self.so
        if not hasattr(so, "aiednn_entry_count"):
            return {}
        so.aiednn_entry_name.restype = C.c_char_p
        so.aiednn_entry_calls_get.restype = C.c_long
        out = {so.aiednn_entry_name(i).decode(): so.aiednn_entry_calls_get(i) for i in range(so.aiednn_entry_count())}
        if reset:
            so.aiednn_entry_calls_reset()
        return out

    def status(self, st) -> str:
        return self.so.aiednn_status_str(st).decode()

    def create(self, fn: str, *args):
        prim = C.c_void_p()
        st = getattr(self.so, fn)(C.byref(prim), *args)
        return prim, st

    def execute(self, prim, bufs: dict):
        """bufs: {arg id: numpy array}"""
        n = len(bufs)
        arr = (ExecArg * n)()
        for i, (a, x) in enumerate(bufs.items()):
            arr[i].arg = a
            arr[i].data = x.ctypes.data
        return self.so.aiednn_primitive_execute_args(prim, n, arr)

    def impl(self, prim) -> str:
        return self.so.aiednn_primitive_impl_info(prim).decode()

    def destroy(self, prim):
        self.so.aiednn_primitive_destroy(prim)

    def set_filter(self, prefix):
        self.so.aiednn_set_impl_filter(prefix.encode() if prefix else None)

    def call(self, fn: str, create_args: list, bufs: dict, out_key: int):
        """create + execute + destroy; returns (out array, impl name)."""
        prim, st = self.create(fn, *create_args)
        if st != 0:
            raise RuntimeError(f"{fn}: {self.status(st)}")
        impl = self.impl(prim)
        st = self.execute(prim, bufs)
        self.destroy(prim)
        if st != 0:
            raise RuntimeError(f"{fn} execute: {self.status(st)}")
        return bufs[out_key], impl


def f(x) -> C.c_float:
    return C.c_float(x)


def pack4(v: np.ndarray) -> np.ndarray:
    """int values in [-8, 7] / [0, 15] -> aiednn s4 / u4 bytes (element 2i in the low nibble)"""
    flat = (np.asarray(v).reshape(-1).astype(np.int64) & 0xF).astype(np.uint8)
    if flat.size % 2:
        flat = np.append(flat, np.uint8(0))
    return (flat[0::2] | (flat[1::2] << 4)).astype(np.uint8)


def unpack4(b: np.ndarray, n: int, signed: bool) -> np.ndarray:
    lo, hi = b & 0xF, b >> 4
    v = np.stack([lo, hi], 1).reshape(-1)[:n].astype(np.int64)
    return np.where(v >= 8, v - 16, v) if signed else v
