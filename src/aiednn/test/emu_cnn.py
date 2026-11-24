###############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
"""x86 emulation harness for the aiednn CNN templates (conv2d/*_bf16.cc, gemm/fc_gemm_bf16.cc).

A template is a C++ ``template <...> __global__`` kernel, read verbatim with :func:`aiednn.template` and launched
as ``<kernel><targs><<<mesh>>>`` (:func:`launch_callee`), wrapped in the kemu kernel model
(``aiegraph.kemu._PRELUDE``: aie::port blocks, mesh launch, ASan-checked block bounds) and
``src/mlir/runtime/aie_api_x86emu.h`` (bf16 x bf16 -> accfloat mmul), compiled with g++ and run
on the layer's tensors. The layer -- shape AND weights -- comes from the real ONNX model: a
missing model, compiler or failed build is a test FAILURE, never a skip.
"""
from __future__ import annotations

import functools
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parents[2]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
import aiednn  # noqa: E402
from aiegraph.kemu import _LAUNCH, _PRELUDE  # noqa: E402

REPO = SRC.parent
# The real models (key -> ONNX path) live with the examples, not in the library: the tests only name the layer kind.
MODELS = {k: Path(v) for k, v in json.loads((REPO / "example" / "aiednn" / "emu_models.json").read_text()).items()
          if not k.startswith("_")}
COS_MIN = 0.99999


# ---------------------------------------------------------------- bf16
def bf16_bits(a: np.ndarray) -> np.ndarray:
    """fp32 -> bf16 bit pattern, round to nearest even (as aie_api_x86emu.h / the AIE)."""
    u = np.ascontiguousarray(a, dtype=np.float32).view(np.uint32).astype(np.uint64)
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)


def bf16_float(bits: np.ndarray) -> np.ndarray:
    return (bits.astype(np.uint32) << 16).view(np.float32)


def bf16(a: np.ndarray) -> np.ndarray:
    return bf16_float(bf16_bits(a))


def cos(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.astype(np.float64).ravel(), b.astype(np.float64).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


# ---------------------------------------------------------------- the ONNX layer
@functools.lru_cache(maxsize=None)
def _model(name: str):
    import onnx
    from onnx import numpy_helper, shape_inference
    p = MODELS[name]
    assert p.is_file(), f"{name}: {p} is missing (the test needs the real model)"
    m = shape_inference.infer_shapes(onnx.load(str(p)))
    g = m.graph
    shp = {v.name: tuple(d.dim_value for d in v.type.tensor_type.shape.dim)
           for v in list(g.value_info) + list(g.input) + list(g.output)}
    init = {i.name: numpy_helper.to_array(i) for i in g.initializer}
    for i, a in init.items():
        shp[i] = a.shape
    cons = {}
    for n in g.node:
        for i in n.input:
            cons.setdefault(i, []).append(n.op_type)
    return g, shp, init, cons


def layer(model: str, op: str, x_shape, w_shape, strides=(), group=1):
    """The first ``op`` node of *model* with this input / weight shape, stride and group:
    ``(weight, bias, attrs, consumer op types)``. Asserts it exists (shape from the real ONNX)."""
    import onnx
    g, shp, init, cons = _model(model)
    for n in g.node:
        if n.op_type != op:
            continue
        a = {x.name: onnx.helper.get_attribute_value(x) for x in n.attribute}
        if (shp.get(n.input[0]) == tuple(x_shape) and shp.get(n.input[1]) == tuple(w_shape)
                and tuple(a.get("strides", ())) == tuple(strides) and a.get("group", 1) == group):
            assert n.input[1] in init, f"{n.name}: weight is not an initializer"
            b = init[n.input[2]] if len(n.input) > 2 and n.input[2] else np.zeros(w_shape[0], np.float32)
            return init[n.input[1]].astype(np.float32), b.astype(np.float32), a, cons.get(n.output[0], [])
    raise AssertionError(f"{model}: no {op} with x {x_shape} w {w_shape} strides {strides} group {group}")


# ---------------------------------------------------------------- numpy references (fp64 on bf16-rounded inputs)
def act_np(z: np.ndarray, act: int) -> np.ndarray:
    if act == 1:
        return np.maximum(z, 0)
    if act == 2:
        return z / (1 + np.exp(-z))
    return z


def conv_np(x, w, b, s, p, group=1, act=0):
    """x [C][H][W], w [CO][C/group][KH][KW] -> [CO][OH][OW]."""
    x, w = x.astype(np.float64), w.astype(np.float64)
    C, H, W = x.shape
    CO, CG, KH, KW = w.shape
    xp = np.pad(x, ((0, 0), (p, p), (p, p)))
    OH, OW = (H + 2 * p - KH) // s + 1, (W + 2 * p - KW) // s + 1
    cols = np.lib.stride_tricks.sliding_window_view(xp, (KH, KW), axis=(1, 2))[:, ::s, ::s][:, :OH, :OW]
    if group == 1:  # [C][OH][OW][KH][KW] x [CO][C][KH][KW]
        z = np.einsum("chwij,ocij->ohw", cols, w, optimize=True)
    else:           # depthwise
        assert group == C == CO and CG == 1
        z = np.einsum("chwij,cij->chw", cols, w[:, 0], optimize=True)
    return act_np(z + b.astype(np.float64)[:, None, None], act)


# ---------------------------------------------------------------- x86 emulation
_MAIN = r"""
static void kemu_rd(const char *f, void *p, size_t n) {
    FILE *fp = fopen(f, "rb");
    if (!fp || fread(p, 1, n, fp) != n) { printf("KEMU FAIL cannot read %s\n", f); exit(2); }
    fclose(fp);
}
int main() {
    void *a0 = __Runtime_Alloc(@N0@), *a1 = __Runtime_Alloc(@N1@), *a2 = __Runtime_Alloc(@N2@), *a3 = __Runtime_Alloc(@N3@);
    kemu_rd("a0.bin", a0, @N0@); kemu_rd("a1.bin", a1, @N1@); kemu_rd("a2.bin", a2, @N2@);
    memset(a3, 0xFF, @N3@); // NaN: an output element no core wrote stays NaN
    aieMesh mesh{@MR@, @MC@};
    @KNAME@<<<mesh>>>(a0, a1, a2, a3);
    FILE *o = fopen("a3.bin", "wb");
    fwrite(a3, 1, @N3@, o);
    fclose(o);
    free(a0); free(a1); free(a2); free(a3);
    printf("KEMU %s %d kernel-model error(s)\n", kemu::g_err ? "FAIL" : "DONE", kemu::g_err);
    return kemu::g_err != 0;
}
"""


_TPL_HEAD = re.compile(r"template\s*<([^<>]*)>\s*__global__\s+void\s+(\w+)\s*\(")


def template_head(template: str):
    """``(kernel name, [template parameter names])`` of the aiednn template kernel *template*."""
    m = _TPL_HEAD.search(aiednn.template(template))
    assert m, f"{template}: no 'template <...> __global__ void <name>(' kernel"
    names = [d.split()[-1].split("=")[0].strip() for d in m.group(1).split(",") if d.strip()]
    return m.group(2), names


def launch_callee(template: str, params: dict) -> str:
    """``<kernel><v0, v1, ...>``: the template arguments of *params* in the order of the template header
    (every parameter given, nothing extra)."""
    name, names = template_head(template)
    assert set(params) == set(names), f"{template}: missing {sorted(set(names) - set(params))} unknown {sorted(set(params) - set(names))}"
    return name + "<" + ", ".join(str(int(params[n])) for n in names) + ">"


def emulate(template: str, params: dict, inputs, out_shape, out_float: bool, mesh=(2, 2), timeout=900) -> np.ndarray:
    """Run aiednn *template* (the instance with template arguments *params*) on x86 for
    *inputs* (3 arrays: uint16 bf16 bits or float32), return the output (fp32 values)."""
    gxx = shutil.which("g++")
    assert gxx, "no host g++ on PATH"
    src = aiednn.template(template)
    callee = launch_callee(template, params)
    with tempfile.TemporaryDirectory(prefix="aiednn_emu_") as tmp:
        t = Path(tmp)
        (t / "kemu_prelude.h").write_text(_PRELUDE)
        (t / "kernel.cc").write_text(src)
        n_out = int(np.prod(out_shape)) * (4 if out_float else 2)
        sizes = {}
        for i, a in enumerate(inputs):
            a = np.ascontiguousarray(a)
            assert a.dtype in (np.uint16, np.float32), a.dtype
            (t / f"a{i}.bin").write_bytes(a.tobytes())
            sizes[f"N{i}"] = a.nbytes
        main = _MAIN
        for k, v in {**sizes, "N3": n_out, "MR": mesh[0], "MC": mesh[1], "KNAME": callee}.items():
            main = main.replace(f"@{k}@", str(v))
        harness = '#include "kemu_prelude.h"\n#include "kernel.cc"\n' + main
        (t / "harness.cc").write_text(_LAUNCH.sub(r"kemu::launch(\2, \1, ", harness))
        cmd = [gxx, "-std=c++20", "-w", "-O1", "-g", "-fsanitize=address", "-fno-omit-frame-pointer",
               f"-I{t}", f"-I{REPO / 'src/mlir/runtime'}", str(t / "harness.cc"), "-o", str(t / "emu")]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        assert r.returncode == 0, f"x86 build of {template} failed:\n{(r.stdout + r.stderr)[-4000:]}"
        r = subprocess.run([str(t / "emu")], capture_output=True, text=True, timeout=timeout, cwd=t)
        out = r.stdout + r.stderr
        assert r.returncode == 0 and "KEMU DONE 0 kernel-model" in out, f"x86 run of {template} failed:\n{out[-4000:]}"
        raw = (t / "a3.bin").read_bytes()
    assert len(raw) == n_out
    y = np.frombuffer(raw, np.float32 if out_float else np.uint16)
    y = y.copy() if out_float else bf16_float(y)
    assert not np.isnan(y).any(), f"{template}: {int(np.isnan(y).sum())} output elements never written"
    return y.reshape(out_shape)


# ---------------------------------------------------------------- tiling choices (what aiegraph would pick)
def largest_divisor(n: int, limit: int, mult: int = 1) -> int:
    best = None
    for d in range(1, n + 1):
        if n % d == 0 and d % mult == 0 and d <= limit:
            best = d
    assert best, (n, limit, mult)
    return best


def tco(co: int) -> int:
    return 16 if co % 16 == 0 else 8 if co % 8 == 0 else 4
