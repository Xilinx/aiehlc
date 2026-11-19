###############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
"""The aiednn MLIR dialect as the whole-model IR (src/mlir/mlirfront/frontend/aiednn).

    text, store = to_mlir(g)                  # AGraph -> aiednn MLIR + constant store
    text = opt(text, "--canonicalize")        # aiednn-opt: verify / run passes
    g2 = from_mlir(text, store)               # aiednn-opt --aiednn-export-plan -> AGraph

One op per aiednn primitive (``aiednn.conv``, ``aiednn.reorder``, ...), SSA
tensors inside a ``func.func``; constants are ``aiednn.const {index = N}`` into
*store* (weights never enter the IR); each op's location is its fx node name;
fused epilogues are ``output_scales`` + ``#aiednn.post_op<...>`` attributes.
Floats are written with 9 significant digits and come back as their f32 bits,
so the round trip is bit-exact (test_dnn_mlir.py). doc/design/aiednn.md §11.
"""

from __future__ import annotations

import json
import os
import shutil
import struct
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from ._env import repo_root
from .errors import AiegraphError
from .dnn_torch import AGraph

_MT = {"f32": "f32", "s8": "si8", "u8": "ui8", "s32": "si32"}
_DT = {v: k for k, v in _MT.items()}

# aiednn create function -> dialect op (the _attr variant maps to the same op)
_OP = {"aiednn_convolution_forward_create": "conv", "aiednn_deconvolution_forward_create": "deconv",
       "aiednn_matmul_forward_create": "matmul", "aiednn_inner_product_forward_create": "inner_product",
       "aiednn_eltwise_forward_create": "eltwise", "aiednn_binary_create": "binary",
       "aiednn_reorder_create": "reorder", "aiednn_prelu_forward_create": "prelu",
       "aiednn_reduction_create": "reduction", "aiednn_pooling_forward_create": "pooling",
       "aiednn_resampling_forward_create": "resampling", "aiednn_lrn_forward_create": "lrn",
       "aiednn_softmax_forward_create": "softmax", "aiednn_shuffle_forward_create": "shuffle",
       "aiednn_batch_normalization_forward_inference_create": "batch_norm",
       "aiednn_layer_normalization_forward_inference_create": "layer_norm",
       "aiednn_group_normalization_forward_inference_create": "group_norm",
       "aiednn_concat_create": "concat", "aiednn_sdpa_forward_create": "sdpa", "aiednn_slice_create": "slice",
       "aiednn_rope_create": "rope", "aiednn_custom_create": "custom"}
_FN = {v: k for k, v in _OP.items()}
_GEMM = ("conv", "deconv", "matmul", "inner_product")


def _ttype(t) -> str:
    return "tensor<" + "".join(f"{d}x" for d in t.shape) + _MT[t.dt] + ">"


def _f(v) -> str:
    return f"{float(np.float32(v)):.9g}"


def _f32(v) -> str:
    if not np.isfinite(np.float32(v)):              # inf / nan: the f32 bit pattern
        return f"0x{struct.unpack('<I', struct.pack('<f', np.float32(v)))[0]:08X} : f32"
    s = _f(v)
    return (s if any(c in s for c in ".eEn") else s + ".0") + " : f32"


def _i64s(v) -> str:
    return "array<i64: " + ", ".join(str(int(x)) for x in v) + ">"


def _by(items, kind):
    return [i for i in items if i[0] == kind]


# ── AGraph -> MLIR ─────────────────────────────────────────────────────────

class _Emitter:
    def __init__(self, g: AGraph):
        self.g, self.lines, self.name, self.store = g, [], {}, []

    def val(self, tid) -> str:
        """SSA name of tensor *tid*, materializing a const / view on first use"""
        if tid in self.name:
            return self.name[tid]
        t = self.g.tensors[tid]
        if t.kind == "const":
            self.store.append(np.ascontiguousarray(t.data))
            self.lines.append(f'    %t{tid} = "aiednn.const"() <{{index = {len(self.store) - 1} : i64}}> : () -> {_ttype(t)}')
        elif t.base is not None:
            src = self.val(t.base)
            self.lines.append(f'    %t{tid} = "aiednn.view"({src}) : ({_ttype(self.g.tensors[t.base])}) -> {_ttype(t)}')
        else:
            raise AiegraphError("dnn_mlir", f"tensor {tid} is used before any op writes it")
        self.name[tid] = f"%t{tid}"
        return self.name[tid]

    def op(self, o) -> None:
        kind = _OP.get(o.fn.replace("_create_attr", "_create"))
        if kind is None:
            raise AiegraphError("dnn_mlir", f"{o.name}: {o.fn} has no aiednn dialect op")
        if self.g.tensors[o.out].base is not None:
            raise AiegraphError("dnn_mlir", f"{o.name}: writes into a view (not SSA)")
        ex = dict((a if isinstance(a, str) else a[0] + str(a[1]), t) for a, t in o.exec)
        operands, attrs, seg = getattr(self, "_" + kind)(o, ex)
        names = [self.val(t) for t in operands]
        types = ", ".join(_ttype(self.g.tensors[t]) for t in operands)
        a = dict(attrs)
        if seg is not None:
            a["operandSegmentSizes"] = "array<i32: " + ", ".join(map(str, seg)) + ">"
        attr_s = (" <{" + ", ".join(f"{k} = {v}" for k, v in a.items()) + "}>") if a else ""   # properties
        res = f"%t{o.out}"
        self.name[o.out] = res
        self.lines.append(f'    {res} = "aiednn.{kind}"({", ".join(names)}){attr_s} : ({types}) -> '
                          f'{_ttype(self.g.tensors[o.out])} loc("{o.name}")')

    # one method per op: (operand tids, attrs {name: mlir text}, operandSegmentSizes or None)
    def _gemm(self, o, ex, geom):
        cr = o.create
        at = next((i[1] for i in cr if i[0] == "attr"), None)
        posts = [t for k, t in sorted((int(k[4:]), t) for k, t in ex.items() if k.startswith("post"))]
        ops = [ex["SRC"], ex["WEIGHTS"]] + ([ex["BIAS"]] if "BIAS" in ex else []) + posts
        attrs = dict(geom)
        if at is not None:
            if at.get("scales") is not None:
                attrs["output_scales"] = "array<f32: " + ", ".join(_f(s) for s in np.ravel(at["scales"])) + ">"
            if at.get("post"):
                ab = lambda p: (p[2], p[3]) if p[0] == "eltwise" else (p[3], p[4]) if len(p) > 3 else (0.0, 0.0)  # noqa: E731
                pos = [f'#aiednn.post_op<"{p[0]}", "{p[1]}", {_f32(ab(p)[0])}, {_f32(ab(p)[1])}>' for p in at["post"]]
                attrs["post_ops"] = "[" + ", ".join(pos) + "]"
            if at.get("dst_scale"):
                attrs["dst_scale"] = _f32(at["dst_scale"])
                attrs["dst_zero_point"] = f"{int(at.get('dst_zp', 0))} : i64"
        return ops, attrs, [1, 1, int("BIAS" in ex), len(posts)]

    def _conv(self, o, ex):
        d = _by(o.create, "conv")[0][1]
        return self._gemm(o, ex, {"strides": _i64s(d["strides"]), "pad_l": _i64s(d["pl"]), "pad_r": _i64s(d["pr"]),
                                  "dilations": _i64s(d["dil"]), "groups": f"{int(d['groups'])} : i64"})

    _deconv = _conv

    def _matmul(self, o, ex):
        return self._gemm(o, ex, {"weights_transposed": "true" if _by(o.create, "int")[0][1] else "false"})

    def _inner_product(self, o, ex):
        return self._gemm(o, ex, {})

    @staticmethod
    def _alg(o, prefix):
        return '"' + _by(o.create, "alg")[0][1][len(prefix):] + '"'

    def _eltwise(self, o, ex):
        f = _by(o.create, "float")
        return [ex["SRC"]], {"alg": self._alg(o, "eltwise_"), "alpha": _f32(f[0][1]), "beta": _f32(f[1][1])}, None

    def _binary(self, o, ex):
        return [ex["SRC"], ex["SRC_1"]], {"alg": self._alg(o, "binary_")}, None

    def _reorder(self, o, ex):
        a = {"scale": _f32(_by(o.create, "float")[0][1]), "zero_point": f"{_by(o.create, 'int')[0][1]} : i64"}
        perm = _by(o.create, "perm")
        if perm:
            a["perm"] = "array<i32: " + ", ".join(map(str, perm[0][1])) + ">"
        return [ex["SRC"]], a, None

    def _prelu(self, o, ex):
        return [ex["SRC"], ex["WEIGHTS"]], {}, None

    def _reduction(self, o, ex):
        f = _by(o.create, "float")
        return [ex["SRC"]], {"alg": self._alg(o, "reduction_"), "p": _f32(f[0][1]), "eps": _f32(f[1][1])}, None

    def _pooling(self, o, ex):
        d = _by(o.create, "pool")[0][1]
        return [ex["SRC"]], {"alg": self._alg(o, "pooling_"), "kernel": _i64s(d["kernel"]),
                             "strides": _i64s(d["strides"]), "pad_l": _i64s(d["pl"]), "pad_r": _i64s(d["pr"]),
                             "dilations": _i64s(d["dil"])}, None

    def _resampling(self, o, ex):
        return [ex["SRC"]], {"alg": self._alg(o, "resampling_")}, None

    def _lrn(self, o, ex):
        f = _by(o.create, "float")
        return [ex["SRC"]], {"alg": self._alg(o, "lrn_"), "local_size": f"{_by(o.create, 'i64')[0][1]} : i64",
                             "alpha": _f32(f[0][1]), "beta": _f32(f[1][1]), "k": _f32(f[2][1])}, None

    def _softmax(self, o, ex):
        return [ex["SRC"]], {"alg": self._alg(o, "softmax_"), "axis": f"{_by(o.create, 'int')[0][1]} : i64"}, None

    def _slice(self, o, ex):
        arrs = _by(o.create, "i64s")
        return [ex["SRC"]], {"offsets": _i64s(arrs[0][1]), "steps": _i64s(arrs[1][1])}, None

    def _rope(self, o, ex):
        a = {"interleaved": f"{_by(o.create, 'int')[0][1]} : i64"}
        at = next((i[1] for i in o.create if i[0] == "attr"), None)
        if at is not None and at.get("dst_scale"):          # folded quantize (create_attr)
            a["dst_scale"] = _f32(at["dst_scale"])
            a["dst_zero_point"] = f"{int(at.get('dst_zp', 0))} : i64"
        return [ex["SRC"], ex["ROPE_COS"], ex["ROPE_SIN"]], a, None

    def _shuffle(self, o, ex):
        return [ex["SRC"]], {"axis": f"{_by(o.create, 'int')[0][1]} : i64",
                             "group_size": f"{_by(o.create, 'i64')[0][1]} : i64"}, None

    def _norm_common(self, o, ex, lead):
        flags = _by(o.create, "uint")[0][1]
        ops = lead + [ex[k] for k in ("SCALE", "SHIFT") if k in ex]
        seg = [len(lead), int("SCALE" in ex), int("SHIFT" in ex)]
        return ops, {"eps": _f32(_by(o.create, "float")[0][1]), "flags": f"{flags} : i64"}, seg

    def _batch_norm(self, o, ex):
        ops, a, seg = self._norm_common(o, ex, [ex["SRC"], ex["MEAN"], ex["VARIANCE"]])
        at = next((i[1] for i in o.create if i[0] == "attr"), None)
        if at is not None and at.get("dst_scale"):      # folded quantize (create_attr)
            a["dst_scale"] = _f32(at["dst_scale"])
            a["dst_zero_point"] = f"{int(at.get('dst_zp', 0))} : i64"
        return ops, a, [1, 1, 1] + seg[1:]

    def _layer_norm(self, o, ex):
        ops, a, seg = self._norm_common(o, ex, [ex["SRC"]])
        at = next((i[1] for i in o.create if i[0] == "attr"), None)
        if at is not None and at.get("dst_scale"):      # folded quantize (create_attr)
            a["dst_scale"] = _f32(at["dst_scale"])
            a["dst_zero_point"] = f"{int(at.get('dst_zp', 0))} : i64"
        return ops, a, seg

    def _group_norm(self, o, ex):
        ops, a, seg = self._norm_common(o, ex, [ex["SRC"]])
        a["groups"] = f"{_by(o.create, 'i64')[0][1]} : i64"
        return ops, a, seg

    def _custom(self, o, ex):
        """a framework custom op run on the host: its sources in order, the op's qualified name"""
        srcs = [t for k, t in sorted((int(k[5:]), t) for k, t in ex.items() if k.startswith("multi"))]
        return srcs, {"name": json.dumps(next(i[1] for i in o.create if i[0] == "custom"))}, None

    def _concat(self, o, ex):
        srcs = [t for k, t in sorted((int(k[5:]), t) for k, t in ex.items() if k.startswith("multi"))]
        return srcs, {"axis": f"{_by(o.create, 'int')[1][1]} : i64"}, None

    def _sdpa(self, o, ex):
        cz = int(_by(o.create, "int")[0][1])         # bit 0 causal, bits 1.. AIEDNN_SDPA_*_BSHD
        a = {"scale": _f32(_by(o.create, "float")[0][1]), "causal": "true" if cz & 1 else "false"}
        if cz >> 1:
            a["layout"] = f"{cz >> 1} : i64"
        at = next((i[1] for i in o.create if i[0] == "attr"), None)
        if at is not None and at.get("dst_scale"):          # folded quantize (create_attr)
            a["dst_scale"] = _f32(at["dst_scale"])
            a["dst_zero_point"] = f"{int(at.get('dst_zp', 0))} : i64"
        if at is not None and at.get("src_scales"):         # 8-bit q / k / v (int8 attention)
            sc = list(at["src_scales"]) + list(at.get("src_scales_23") or [0.0, 0.0])
            a["src_scales"] = "array<f32: " + ", ".join(t if any(c in t for c in ".en") else t + ".0"
                                                         for t in (_f(v) for v in sc)) + ">"   # 0 -> 0.0
        if at is not None and at.get("flags"):
            a["flags"] = f"{int(at['flags'])} : i64"
        return [ex["SRC"], ex["SRC_1"], ex["SRC_2"]], a, None


def to_mlir(g: AGraph) -> tuple[str, list]:
    """aiednn MLIR text of *g* and its constant store (index -> ndarray)"""
    e = _Emitter(g)
    e.name[g.input] = f"%t{g.input}"
    for o in g.ops:
        e.op(o)
    out = e.val(g.output)
    tin, tout = g.tensors[g.input], g.tensors[g.output]
    head = (f'module attributes {{aiednn.mode = "{g.mode}"}} {{\n'
            f"  func.func @{_sym(g.name)}(%t{g.input}: {_ttype(tin)}) -> {_ttype(tout)} {{")
    return "\n".join([head] + e.lines + [f"    func.return {out} : {_ttype(tout)}", "  }", "}", ""]), e.store


def _sym(name: str) -> str:
    s = "".join(c if c.isalnum() or c == "_" else "_" for c in name)
    return s if s and not s[0].isdigit() else "m_" + s


# ── aiednn-opt ─────────────────────────────────────────────────────────────

def aiednn_opt_path() -> Path | None:
    """$AIEDNN_OPT, else build/aiednn/aiednn-opt under the repo, else PATH"""
    cands = [os.environ.get("AIEDNN_OPT"), repo_root() / "build" / "aiednn" / "aiednn-opt", shutil.which("aiednn-opt")]
    for c in cands:
        if c and Path(c).is_file() and os.access(c, os.X_OK):
            return Path(c)
    return None


def opt(text: str, *args: str) -> str:
    """run aiednn-opt over *text* with *args* (passes); returns the printed IR"""
    exe = aiednn_opt_path()
    if exe is None:
        raise AiegraphError("dnn_mlir", "aiednn-opt not built",
                            hint="source script/llvm_env.sh; cmake -G Ninja -S src/mlir/mlirfront/frontend/aiednn "
                                 "-B build/aiednn && ninja -C build/aiednn (or set AIEDNN_OPT)")
    r = subprocess.run([str(exe), "-", "--mlir-print-debuginfo", *args], input=text, capture_output=True, text=True)
    if r.returncode != 0:
        raise AiegraphError("dnn_mlir", "aiednn-opt failed:\n" + r.stderr[-4000:])
    return r.stdout


def export_plan(text: str, passes=()) -> dict:
    """run *passes* and export the plan in ONE aiednn-opt process: the printed IR is
    not exact for f32 arrays (MLIR prints them in short decimal form), the plan is"""
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "plan.json"
        opt(text, *passes, f"--aiednn-export-plan=out={p}")
        return json.loads(p.read_text())


# ── plan -> AGraph ─────────────────────────────────────────────────────────

def _fl(v) -> float:
    return struct.unpack("<f", struct.pack("<I", v["f32"]))[0]


def _geom(a) -> dict:
    return dict(strides=tuple(a["strides"]), pl=tuple(a["pad_l"]), pr=tuple(a["pad_r"]), dil=tuple(a["dilations"]))


def from_mlir(text: str, store: list, name: str | None = None, mode: str | None = None, passes=()) -> AGraph:
    """the executable AGraph of the (first) func in aiednn MLIR *text*, after *passes*
    (aiednn-opt flags, e.g. "--aiednn-fuse-epilogue")"""
    plan = export_plan(text, passes)
    fn = plan["funcs"][0]
    if mode is None:
        mode = "int8" if 'aiednn.mode = "int8"' in text else "f32"
    return from_plan(fn, store, name or fn["name"], mode)


def from_plan(fn: dict, store: list, name: str, mode: str) -> AGraph:
    g = AGraph(name=name, mode=mode)
    vals = {v["id"]: v for v in fn["values"]}
    tid = {}

    def new(vid, **kw):
        v = vals[vid]
        tid[vid] = g.new(tuple(v["shape"]), _DT.get(v["dtype"], v["dtype"]), **kw)
        return tid[vid]

    (arg,) = fn["args"]
    g.input = new(arg, kind="input")
    for o in fn["ops"]:
        kind, a, opnds, (res,) = o["op"].split(".", 1)[1], o["attrs"], [tid.get(i) for i in o["operands"]], o["results"]
        if kind == "const":
            arr = store[a["index"]]
            v = vals[res]
            tid[res] = g.const(arr.reshape(tuple(v["shape"])), v["dtype"])
            continue
        if kind == "view":
            tid[res] = g.alias(opnds[0], tuple(vals[res]["shape"]))
            continue
        y = new(res)
        if "aiednn.offset" in a:                          # --aiednn-plan-memory
            g.plan = {} if g.plan is None else g.plan
            g.plan[y] = int(a.pop("aiednn.offset"))
        a.pop("aiednn.inplace", None)
        create, ex = _rebuild(kind, a, opnds, o.get("segments"), y, g)
        fnname = _FN[kind]
        if (kind in _GEMM or kind in ("batch_norm", "layer_norm", "sdpa", "rope")) and any(c[0] == "attr" for c in create):
            fnname = fnname.replace("_create", "_create_attr")
        g.op(o["loc"], fnname, create, ex, y)
        for n in o.get("locs", [])[1:]:                 # source ops a fusion pass folded into it
            g.nodes.setdefault(n, y)
    (r,) = fn["results"]
    g.output = tid[r]
    if g.plan is not None:
        g.arena = int(fn.get("attrs", {}).get("aiednn.arena_bytes", 0))
        g.naive_bytes = int(fn.get("attrs", {}).get("aiednn.naive_bytes", 0))
    return g


def _segments(opnds, seg):
    out, i = [], 0
    for n in seg:
        out.append(opnds[i:i + n])
        i += n
    return out


def _rebuild(kind, a, opnds, seg, y, g):  # noqa: C901 -- one branch per op, mirrors _Emitter
    md = lambda t: ("md", t)  # noqa: E731
    if kind in _GEMM:
        src, wt, bias, posts = _segments(opnds, seg)
        b = bias[0] if bias else None
        create = [md(src[0]), md(wt[0]), md(b) if b is not None else ("null",), md(y)]
        if kind in ("conv", "deconv"):
            create.append(("conv", dict(_geom(a), groups=a["groups"])))
        elif kind == "matmul":
            create.append(("int", int(bool(a["weights_transposed"]))))
        ex = [("SRC", src[0]), ("WEIGHTS", wt[0]), ("DST", y)] + ([("BIAS", b)] if b is not None else [])
        if "output_scales" in a or "post_ops" in a or "dst_scale" in a:
            sc = np.array([_fl(s) for s in a["output_scales"]], np.float32) if "output_scales" in a else None
            post, k = [], 0
            for p in a.get("post_ops", []):
                if p["kind"] == "eltwise":
                    post.append(("eltwise", p["alg"], _fl(p["alpha"]), _fl(p["beta"])))
                else:
                    rs, rz = _fl(p["alpha"]), _fl(p["beta"])  # an s8 / u8 src1's scale / zero point
                    post.append(("binary", p["alg"], posts[k]) + ((rs, rz) if rs or rz else ()))
                    k += 1
            at = dict(scales=sc, post=post)
            if "dst_scale" in a:                       # --aiednn-fold-requant: integer dst
                at.update(dst_scale=_fl(a["dst_scale"]), dst_zp=int(a.get("dst_zero_point", 0)))
            create.append(("attr", at))
            ex += [(("post", j), p[2]) for j, p in enumerate(post) if p[0] == "binary"]
        return create, ex
    x = opnds[0] if opnds else None
    simple = {"SRC": x, "DST": y}
    if kind == "eltwise":
        return ([("alg", "eltwise_" + a["alg"]), ("float", _fl(a["alpha"])), ("float", _fl(a["beta"])), md(x), md(y)],
                list(simple.items()))
    if kind == "binary":
        return ([("alg", "binary_" + a["alg"]), md(opnds[0]), md(opnds[1]), ("null",), md(y)],
                [("SRC", opnds[0]), ("SRC_1", opnds[1]), ("DST", y)])
    if kind == "reorder":
        perm = ("perm", list(a["perm"])) if "perm" in a else ("null",)
        return [md(x), md(y), perm, ("float", _fl(a["scale"])), ("int", a["zero_point"])], list(simple.items())
    if kind == "prelu":
        return [md(x), md(opnds[1]), md(y)], [("SRC", x), ("WEIGHTS", opnds[1]), ("DST", y)]
    if kind == "reduction":
        return ([("alg", "reduction_" + a["alg"]), ("float", _fl(a["p"])), ("float", _fl(a["eps"])), md(x), md(y)],
                list(simple.items()))
    if kind == "pooling":
        pp = dict(_geom(a), kernel=tuple(a["kernel"]))
        return [("alg", "pooling_" + a["alg"]), md(x), md(y), ("pool", pp)], list(simple.items())
    if kind == "resampling":
        return [("alg", "resampling_" + a["alg"]), md(x), md(y)], list(simple.items())
    if kind == "lrn":
        return ([("alg", "lrn_" + a["alg"]), ("i64", a["local_size"]), ("float", _fl(a["alpha"])),
                 ("float", _fl(a["beta"])), ("float", _fl(a["k"])), md(x), md(y)], list(simple.items()))
    if kind == "softmax":
        return [("alg", "softmax_" + a["alg"]), ("int", a["axis"]), md(x), md(y)], list(simple.items())
    if kind == "shuffle":
        return [("int", a["axis"]), ("i64", a["group_size"]), md(x), md(y)], list(simple.items())
    if kind == "slice":
        return [md(x), md(y), ("i64s", list(a["offsets"])), ("i64s", list(a["steps"]))], list(simple.items())
    if kind == "rope":
        at = ([("attr", dict(scales=None, post=[], dst_scale=_fl(a["dst_scale"]),
                             dst_zp=int(a.get("dst_zero_point", 0))))] if "dst_scale" in a else [])
        return ([md(x), md(opnds[1]), md(opnds[2]), md(y), ("int", int(a["interleaved"]))] + at,
                [("SRC", x), ("ROPE_COS", opnds[1]), ("ROPE_SIN", opnds[2]), ("DST", y)])
    if kind in ("batch_norm", "layer_norm", "group_norm"):
        parts = _segments(opnds, seg)
        ex = [("SRC", x), ("DST", y)]
        if kind == "batch_norm":
            ex += [("MEAN", parts[1][0]), ("VARIANCE", parts[2][0])]
            parts = parts[3:]
        else:
            parts = parts[1:]
        ex += [(k, p[0]) for k, p in zip(("SCALE", "SHIFT"), parts) if p]
        create = [md(x), md(y), ("float", _fl(a["eps"])), ("uint", a["flags"])]
        if "dst_scale" in a:                           # --aiednn-fold-norm-quant: s8 / u8 dst
            # flags 1 = AIEDNN_ATTR_TOLERANCE_1CODE: the AIE (fvec) norm -> int8 may differ from ref: by one code
            # at a rounding boundary; the model checks against PyTorch with a tolerance, so take the AIE path
            create.append(("attr", dict(scales=None, post=[], dst_scale=_fl(a["dst_scale"]),
                                        dst_zp=int(a.get("dst_zero_point", 0)), flags=1)))
        if kind == "group_norm":
            create = [("i64", a["groups"])] + create
        return create, ex
    if kind == "custom":
        return ([("int", len(opnds)), ("mds", list(opnds)), md(y), ("custom", a["name"])],
                [(("multi", i), t) for i, t in enumerate(opnds)] + [("DST", y)])
    if kind == "concat":
        return ([("int", len(opnds)), ("mds", list(opnds)), ("int", a["axis"]), md(y)],
                [(("multi", i), t) for i, t in enumerate(opnds)] + [("DST", y)])
    if kind == "sdpa":
        q, k_, v = opnds[:3]
        at = None
        if "dst_scale" in a or "src_scales" in a or "flags" in a:
            at = dict(scales=None, post=[], dst_scale=_fl(a["dst_scale"]) if "dst_scale" in a else 0.0,
                      dst_zp=int(a.get("dst_zero_point", 0)), flags=int(a.get("flags", 0)))
            if "src_scales" in a:
                sc = [_fl(v) for v in a["src_scales"]]
                at.update(src_scales=sc[:2], src_scales_23=sc[2:4])
        return ([md(q), md(k_), md(v), ("null",), md(y), ("float", _fl(a["scale"])),
                 ("int", int(bool(a["causal"])) | (int(a.get("layout", 0)) << 1))]   # layout: AIEDNN_SDPA_*_BSHD
                + ([("attr", at)] if at else []),
                [("SRC", q), ("SRC_1", k_), ("SRC_2", v), ("DST", y)])
    raise AiegraphError("dnn_mlir", f"no rebuild rule for aiednn.{kind}")


#: the default aiednn pipeline: clean up, fuse epilogues, keep activations int8
#: between GEMMs, clean up again (canonicalize folds the reorders the requant
#: folding exposed; cse merges identical quantizations of one tensor), then plan
#: the activation arena (liveness reuse + in-place eltwise)
# quantize-residual before quantize-before-maxpool: once a skip's int8 copy is the max-pool's only
# consumer (resnet's stem pool feeds layer1's conv and its residual), the pool can move to int8
PIPELINE = ("--canonicalize", "--cse", "--aiednn-fuse-norm-relu", "--aiednn-sink-permute", "--aiednn-fuse-epilogue", "--aiednn-quantize-residual", "--cse",
            "--aiednn-quantize-through-views", "--cse",
            "--aiednn-quantize-before-maxpool", "--aiednn-fold-requant", "--aiednn-fold-head-permute",
            "--aiednn-fold-norm-quant",
            "--canonicalize", "--cse",
            "--aiednn-plan-memory")


_GEMM_FNS = ("aiednn_matmul_forward_create", "aiednn_matmul_forward_create_attr")


def _uses(g, tid) -> list:
    """indices of the ops reading tensor *tid* (create descriptors or exec arguments)"""
    return [i for i, op in enumerate(g.ops)
            if any(a[0] == "md" and a[1] == tid for a in op.create if isinstance(a, tuple))
            or any(e[1] == tid for e in op.exec if e[0] not in ("DST",))]


def _slice_range(g, op, n_axis_len):
    """(offset, width) of a step-1 slice along the last axis only, else None"""
    offs = next((a[1] for a in op.create if a[0] == "i64s"), None)
    steps = [a[1] for a in op.create if a[0] == "i64s"][1:2]
    if offs is None or (steps and any(s != 1 for s in steps[0])) or any(offs[:-1]):
        return None
    src, dst = g.tensors[op.exec[0][1]], g.tensors[op.out]
    if tuple(src.shape[:-1]) != tuple(dst.shape[:-1]) or src.shape[-1] != n_axis_len:
        return None
    return int(offs[-1]), int(dst.shape[-1])


def split_gemm_slices(g: AGraph) -> int:
    """A GEMM (constant transposed weights [.., N, K], optional constant bias) whose result only feeds
    slices along its N axis becomes one GEMM per slice, each writing the slice's tensor directly: the
    weight rows, bias and per-channel output scales are cut the same way. Exact (each output channel is
    computed alone), and the slices leave the host (gpt2's c_attn -> q / k / v: 36 slices). Returns the
    number of GEMMs split."""
    n_split = 0
    i = 0
    while i < len(g.ops):
        op = g.ops[i]
        attr = next((c[1] for c in op.create if c[0] == "attr"), None)
        ex = dict((e[0], e[1]) for e in op.exec if isinstance(e[0], str))
        w = g.tensors[ex["WEIGHTS"]] if "WEIGHTS" in ex else None
        ok = (op.fn in _GEMM_FNS and w is not None and w.kind == "const" and w.base is None
              and ("int", 1) in op.create and (attr is None or not attr.get("post"))
              and op.out != g.output)
        users = [u for u in _uses(g, op.out) if u != i] if ok else []   # its own create names its dst
        n = g.tensors[op.out].shape[-1] if ok else 0
        ranges = [_slice_range(g, g.ops[u], n) if g.ops[u].fn == "aiednn_slice_create" else None for u in users]
        if not ok or not users or any(r is None for r in ranges) or "BIAS" in ex and g.tensors[ex["BIAS"]].kind != "const":
            i += 1
            continue
        new_ops = []
        for u, (off, wid) in zip(users, ranges):
            sl = g.ops[u]
            wt = g.const(np.ascontiguousarray(w.data[..., off:off + wid, :]), w.dt)
            exec_ = [("SRC", ex["SRC"]), ("WEIGHTS", wt), ("DST", sl.out)]
            create = []
            for c in op.create:
                if c[0] == "md" and c[1] == ex["WEIGHTS"]:
                    create.append(("md", wt))
                elif c[0] == "md" and c[1] == op.out:
                    create.append(("md", sl.out))
                elif c[0] == "md" and "BIAS" in ex and c[1] == ex["BIAS"]:
                    b = g.tensors[ex["BIAS"]]
                    bt = g.const(np.ascontiguousarray(b.data[..., off:off + wid]), b.dt)
                    create.append(("md", bt))
                    exec_.append(("BIAS", bt))
                elif c[0] == "attr":
                    a2 = dict(c[1])
                    for k, v in c[1].items():
                        if isinstance(v, np.ndarray) and v.ndim == 1 and v.shape[0] == n:
                            a2[k] = np.ascontiguousarray(v[off:off + wid])
                    create.append(("attr", a2))
                else:
                    create.append(c)
            new_ops.append(type(op)(f"{op.name}_{off}", op.fn, create, exec_, sl.out))
        g.ops[i:i + 1] = new_ops                       # the slices' tensors are written in place of the GEMM
        g.ops = [o for o in g.ops if not (o.fn == "aiednn_slice_create" and o.exec[0][1] == op.out)]
        i += len(new_ops)
        n_split += 1
    return n_split


def _producers(g) -> dict:
    return {op.out: i for i, op in enumerate(g.ops)}


def _ex(op) -> dict:
    return {e[0]: e[1] for e in op.exec if isinstance(e[0], str)}


def _alg(op):
    return next((c[1] for c in op.create if c[0] == "alg"), None)


def _const(g, tid) -> bool:
    return g.tensors[tid].kind == "const" and g.tensors[tid].base is None


def _n_users(g, tid) -> int:
    """ops reading tensor *tid* (its producer excluded)"""
    return sum(1 for u in _uses(g, tid) if g.ops[u].out != tid)


def _drop_ops(g, idx):
    idx = set(idx)
    g.ops = [o for k, o in enumerate(g.ops) if k not in idx]


_RMS_CHAIN = (("aiednn_eltwise_forward_create", "eltwise_rsqrt"), ("aiednn_binary_create", "binary_add"),
              ("aiednn_reduction_create", "reduction_mean"), ("aiednn_eltwise_forward_create", "eltwise_pow"))


def _match_rmsnorm(g, prod, i):
    """(ops to drop, x, w, eps) when op *i* is the last mul of an unfused RMSNorm, else None"""
    op = g.ops[i]
    if op.fn != "aiednn_binary_create" or _alg(op) != "binary_mul":
        return None
    e = _ex(op)
    a, w = e["SRC"], e["SRC_1"]
    if not _const(g, w) or a not in prod or any(d != 1 for d in g.tensors[w].shape[:-1]) or _n_users(g, a) != 1:
        return None
    m1 = prod[a]
    if g.ops[m1].fn != "aiednn_binary_create" or _alg(g.ops[m1]) != "binary_mul":
        return None
    x, t = _ex(g.ops[m1])["SRC"], _ex(g.ops[m1])["SRC_1"]
    chain = []
    for fn, alg in _RMS_CHAIN:                     # rsqrt <- add eps <- mean <- pow, each read once
        if _n_users(g, t) != 1:
            return None
        if t not in prod and g.tensors[t].base is not None:
            t = g.root(t)                              # a keepdim view of the reduction's result
        k = prod.get(t)
        if k is None or g.ops[k].fn != fn or _alg(g.ops[k]) != alg:
            return None
        chain.append(k)
        t = _ex(g.ops[k])["SRC"]
    if t != x:
        return None
    eps_t = _ex(g.ops[chain[1]])["SRC_1"]
    fl = [c[1] for c in g.ops[chain[3]].create if c[0] == "float"]
    if not _const(g, eps_t) or g.tensors[eps_t].data.size != 1 or fl[:2] != [1.0, 2.0]:
        return None
    return chain + [m1], x, w, float(g.tensors[eps_t].data.reshape(-1)[0])


def fuse_rmsnorm(g: AGraph) -> int:
    """x * rsqrt(mean(x ** 2) + eps) * w (an unfused RMSNorm: pow, mean, add, rsqrt, mul, mul) -> one layer norm
    with AIEDNN_USE_SCALE | AIEDNN_RMS_NORM; --aiednn-fold-norm-quant then gives it the int8 dst the next GEMM
    reads. Returns the number fused."""
    n = 0
    while True:
        prod = _producers(g)
        hit = next(((i, m) for i in range(len(g.ops)) if (m := _match_rmsnorm(g, prod, i))), None)
        if hit is None:
            return n
        i, (drop, x, w, eps) = hit
        op = g.ops[i]
        w1 = g.const(np.ascontiguousarray(g.tensors[w].data.reshape(-1)), g.tensors[w].dt)
        g.ops[i] = type(op)(op.name, "aiednn_layer_normalization_forward_inference_create",
                            [("md", x), ("md", op.out), ("float", eps), ("uint", 1 | 8)],   # USE_SCALE | RMS_NORM
                            [("SRC", x), ("DST", op.out), ("SCALE", w1)], op.out)
        _drop_ops(g, drop)
        n += 1


def _bin(g, prod, t, alg):
    """the binary op of *alg* producing *t*: (index, src, src_1) or None"""
    k = prod.get(t)
    if k is None or g.ops[k].fn != "aiednn_binary_create" or _alg(g.ops[k]) != alg:
        return None
    e = _ex(g.ops[k])
    return k, e["SRC"], e["SRC_1"]


def _half_slice(g, prod, t, x, upper):
    """index of the slice of *x* producing *t*: its lower (upper=False) or upper half along the last axis"""
    k = prod.get(t)
    if k is None or g.ops[k].fn != "aiednn_slice_create" or _ex(g.ops[k])["SRC"] != x:
        return None
    r = _slice_range(g, g.ops[k], g.tensors[x].shape[-1])
    d = g.tensors[x].shape[-1]
    return k if r == ((d // 2, d - d // 2) if upper else (0, d // 2)) else None


def _match_rope(g, prod, i):
    """(ops to drop, x, cos, sin) when op *i* is x * cos + cat(-x[.., D/2:], x[.., :D/2]) * sin, else None"""
    op = g.ops[i]
    if op.fn != "aiednn_binary_create" or _alg(op) != "binary_add":
        return None
    a, b = _ex(op)["SRC"], _ex(op)["SRC_1"]
    ma, mb = _bin(g, prod, a, "binary_mul"), _bin(g, prod, b, "binary_mul")
    if not ma or not mb or not _const(g, ma[2]) or not _const(g, mb[2]) or _n_users(g, a) != 1 or _n_users(g, b) != 1:
        return None
    x, cos, c, sin = ma[1], ma[2], mb[1], mb[2]
    kc = prod.get(c)
    if kc is None or g.ops[kc].fn != "aiednn_concat_create" or _n_users(g, c) != 1:
        return None
    srcs = [t for k, t in sorted((e[0][1], e[1]) for e in g.ops[kc].exec if isinstance(e[0], tuple))]
    if len(srcs) != 2 or [c_[1] for c_ in g.ops[kc].create if c_[0] == "int"][1:2] != [len(g.tensors[c].shape) - 1]:
        return None
    kn = prod.get(srcs[0])
    if kn is None or g.ops[kn].fn != "aiednn_eltwise_forward_create" or _alg(g.ops[kn]) != "eltwise_neg":
        return None
    hi = _half_slice(g, prod, _ex(g.ops[kn])["SRC"], x, True)
    lo = _half_slice(g, prod, srcs[1], x, False)
    if hi is None or lo is None or any(_n_users(g, t) != 1 for t in (srcs[0], srcs[1], g.ops[hi].out)):
        return None
    return [ma[0], mb[0], kc, kn, hi, lo], x, cos, sin


def _rope_op(op, name, src, cos, sin, dst):
    return type(op)(name, "aiednn_rope_create", [("md", src), ("md", cos), ("md", sin), ("md", dst), ("int", 0)],
                    [("SRC", src), ("ROPE_COS", cos), ("ROPE_SIN", sin), ("DST", dst)], dst)


def fuse_rope(g: AGraph) -> int:
    """the rotate-half rotary embedding chain (mul, slice, neg, slice, concat, mul, add) -> aiednn_rope_create,
    bit-identical (its documented arithmetic order is mul, mul, add). When x is a head permute
    [B, S, H, D] -> [B, H, S, D] read by nothing else, the rope moves in front of it on [B, S, H, D] with cos /
    sin transposed the same way, so --aiednn-fold-head-permute can fold the permute into sdpa. Returns the count."""
    n = 0
    while True:
        prod = _producers(g)
        hit = next(((i, m) for i in range(len(g.ops)) if (m := _match_rope(g, prod, i))), None)
        if hit is None:
            return n
        i, (drop, x, cos, sin) = hit
        op = g.ops[i]
        kp = prod.get(x)
        perm = next((c[1] for c in g.ops[kp].create if c[0] == "perm"), None) if kp is not None else None
        sinkable = (kp is not None and g.ops[kp].fn == "aiednn_reorder_create" and list(perm or []) == [0, 2, 1, 3]
                    and g.tensors[x].dt == "f32" and _n_users(g, x) == 3          # mul(x, cos) + the two slices
                    and all(g.tensors[t].shape[:2] == (1, 1) for t in (cos, sin)))
        if not sinkable:
            g.ops[i] = _rope_op(op, op.name, x, cos, sin, op.out)
            _drop_ops(g, drop)
            n += 1
            continue
        pre = _ex(g.ops[kp])["SRC"]                    # [B, S, H, D]
        cs = [g.const(np.ascontiguousarray(g.tensors[t].data.transpose(0, 2, 1, 3)), g.tensors[t].dt) for t in (cos, sin)]
        r = g.new(g.tensors[pre].shape, "f32")
        permute = g.ops[kp]
        g.ops[i:i + 1] = [_rope_op(op, op.name, pre, cs[0], cs[1], r),
                          type(permute)(permute.name, permute.fn,
                                        [("md", r) if c_[0] == "md" and c_[1] == pre else
                                         ("md", op.out) if c_[0] == "md" and c_[1] == x else c_ for c_ in permute.create],
                                        [("SRC", r), ("DST", op.out)], op.out)]
        _drop_ops(g, drop + [kp])
        n += 1


REWRITES = ("split_gemm_slices", "fuse_rmsnorm", "fuse_rope")   # AGraph rewrites before the MLIR pipeline


def optimize(g: AGraph, passes=PIPELINE, keep_ir: Path | None = None, rewrites=REWRITES) -> AGraph:
    """*g* through the aiednn MLIR pipeline; returns the executable graph.
    *keep_ir*: also write the input and optimized IR there (<name>.in.mlir / .opt.mlir).
    *rewrites*: the AGraph rewrites run first (they need constant values, which the MLIR only references)."""
    for r in rewrites:
        globals()[r](g)
    text, store = to_mlir(g)
    if keep_ir is not None:
        keep_ir = Path(keep_ir)
        keep_ir.mkdir(parents=True, exist_ok=True)
        (keep_ir / f"{g.name}.in.mlir").write_text(text)
        (keep_ir / f"{g.name}.opt.mlir").write_text(opt(text, *passes))
    g2 = from_mlir(text, store, g.name, g.mode, passes=passes)
    g2.tensors[g2.input].data = g.tensors[g.input].data   # the example input travels with the graph (emit_app)
    return g2


def write(g: AGraph, path) -> Path:
    """<path>.aiednn.mlir + <path>.consts.npz"""
    path = Path(path)
    text, store = to_mlir(g)
    path.with_suffix(".aiednn.mlir").write_text(text)
    np.savez(path.with_suffix(".consts.npz"), *store)
    return path.with_suffix(".aiednn.mlir")
