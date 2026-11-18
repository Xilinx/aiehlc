###############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
"""Bit-exact check of the generated AIE BYOC wrapper against TVM itself.

    PYTHONPATH=src python3 src/frontend/tvmrelay/byoc/verify_aie_stem.py [out_dir]

1. Import the int8 QDQ model, partition every conv the pattern accepts (the stem).
2. Emit the wrapper C for it (``aie_codegen.emit_c_for_function``) and build it
   with gcc against ``conv2dstem/ref/conv2dstem_x86.c`` -- the core's
   arithmetic on x86.
3. Run the SAME composite function through TVM on LLVM and compare, element
   for element, on (a) full-range random uint8 input, which exercises every
   pixel >= 128 that a plain uint8->int8 cast would corrupt, and (b) the real
   dog image quantized with the graph's own scale / zero-point.

4. End to end: build the WHOLE network twice on LLVM -- unpartitioned, and
   partitioned with the wrapper + x86 model linked in -- and require the 1000
   logits on the dog image to be identical. This covers what (3) cannot: the
   graph executor really calling the BYOC function, and the re-canonicalized
   CPU remainder of the graph (partitioning changes how the rest is built).

What this proves: the wrapper's constant folding, uint8 shift, padding and
layout transforms are exact, in isolation and in the full network. What it
does not: the on-core DMA path (see ``conv2dstem/ref/verify_pack.c``) and
execution on hardware.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

# The LLVM reference build would otherwise fetch AutoTVM "tophub" tuning logs
# into ~/.tvm -- performance-only, and irrelevant to a correctness reference.
os.environ.setdefault("TOPHUB_LOCATION", "NONE")

import frontend.tvmrelay  # noqa: F401,E402  (onnx.mapping shim, TVM 0.16 check)
import tvm
from tvm import relay
from tvm.contrib import cc, graph_executor

from frontend.tvmrelay import deploy_flow as df, image_input, onnx_ptq as ptq
from frontend.tvmrelay.byoc import aie_byoc, aie_codegen

REPO = Path(__file__).resolve().parents[4]
X86_MODEL = REPO / "src" / "aietensorop" / "conv2dstem" / "ref" / "conv2dstem_x86.c"

_MAIN_C = r"""
#include <stdio.h>
#include <stdint.h>
int %(name)s_wrapper_(uint8_t* in0, uint8_t* out0);
static uint8_t in0[%(n_in)d], out0[%(n_out)d];
int main(int argc, char** argv) {
    FILE* f = fopen(argv[1], "rb"); fread(in0, 1, sizeof in0, f); fclose(f);
    int rc = %(name)s_wrapper_(in0, out0);
    f = fopen(argv[2], "wb"); fwrite(out0, 1, sizeof out0, f); fclose(f);
    return rc;
}
"""


def _aie_function(mod):
    for gv in mod.get_global_vars():
        fn = mod[gv]
        if isinstance(fn, relay.Function) and fn.attrs and "Compiler" in fn.attrs:
            return str(gv.name_hint), fn
    raise RuntimeError("no AIE subgraph was partitioned out")


def _tvm_reference(fn):
    """Run the composite's body (stripped of BYOC attrs) on LLVM -> callable."""
    comp = aie_codegen._find_composite(fn)
    plain = relay.Function(comp.params, comp.body)
    lib = relay.build(tvm.IRModule.from_expr(plain), target="llvm")
    gm = graph_executor.GraphModule(lib["default"](tvm.cpu()))

    def run(x):
        gm.set_input(0, x)
        gm.run()
        return gm.get_output(0).numpy()
    return run


def _real_input() -> np.ndarray:
    """The dog image (same resolver as stage 6), quantized + padded as the graph does."""
    classify = image_input._classify_mod()

    path = classify.resolve(None, classify.DEFAULT_IMAGE_URL, "input_image")
    img = np.asarray(classify.preprocess(path), np.float32).reshape(1, 3, 224, 224)
    scale, zp = 0.0185913, 113   # data_QuantizeLinear in the graph (see IR)
    q = np.clip(np.round(img / np.float32(scale)) + zp, 0, 255).astype(np.uint8)
    return np.pad(q, ((0, 0), (0, 0), (3, 3), (3, 3)), constant_values=zp)


def _run_network(mod, image, workdir: Path, extra_c=None):
    """relay.build on LLVM (+ x86 model for the BYOC C), run on *image* -> logits."""
    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(mod, target="llvm")
    so = workdir / ("net_aie.so" if extra_c else "net_cpu.so")
    if extra_c:
        def fcompile(output, objects, **kw):
            return cc.create_shared(output, list(objects) + [str(extra_c)], **kw)
        lib.export_library(str(so), fcompile=fcompile)
    else:
        lib.export_library(str(so))
    gm = graph_executor.GraphModule(tvm.runtime.load_module(str(so))["default"](tvm.cpu()))
    gm.set_input(0, image)
    gm.run()
    return gm.get_output(0).numpy()


def _float_image() -> np.ndarray:
    classify = image_input._classify_mod()

    path = classify.resolve(None, classify.DEFAULT_IMAGE_URL, "input_image")
    return np.asarray(classify.preprocess(path), np.float32).reshape(1, 3, 224, 224)


def main(out_dir: str = "worklocal/tvmrelay_deploy") -> int:
    out = Path(out_dir)
    qdq = sorted(out.glob("*.int8.qdq.onnx"))
    if not qdq:
        raise FileNotFoundError(f"no *.int8.qdq.onnx in {out} (run deploy_flow first)")
    base, params = df.import_relay(qdq[0], verbose=False)
    base = ptq.to_integer_ops(base, verbose=False)
    mod, info = aie_byoc.partition_for_aie(base, params, convs=None, verbose=True)
    name, fn = _aie_function(mod)
    ref = _tvm_reference(fn)

    in_shape = [int(d) for d in fn.params[0].checked_type.shape]
    n_in, n_out = int(np.prod(in_shape)), 64 * 112 * 112
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        (td / "wrapper.c").write_text(aie_codegen._C_PRELUDE
                                      + aie_codegen.emit_c_for_function(fn, name))
        (td / "main.c").write_text(_MAIN_C % {"name": name, "n_in": n_in, "n_out": n_out})
        exe = td / "stem"
        subprocess.run(["gcc", "-O2", "-o", str(exe), str(td / "wrapper.c"),
                        str(td / "main.c"), str(X86_MODEL)], check=True)

        rng = np.random.default_rng(0)
        cases = {"random uint8 [0,255]":
                 rng.integers(0, 256, size=in_shape, dtype=np.uint8)}
        cases["dog.jpg"] = _real_input()
        bad = 0
        for label, x in cases.items():
            (td / "in.bin").write_bytes(x.tobytes())
            subprocess.run([str(exe), str(td / "in.bin"), str(td / "out.bin")], check=True)
            got = np.frombuffer((td / "out.bin").read_bytes(), np.uint8).reshape(1, 64, 112, 112)
            want = ref(x)
            diff = int((got != want).sum())
            bad += diff
            print(f"  [verify] {label:22s}: {diff} / {want.size} mismatches "
                  f"(nonzero outputs {int((want != 0).sum())})")

        image = _float_image()
        want = _run_network(base, image, td)
        got = _run_network(mod, image, td, extra_c=X86_MODEL)
        diff = int((got != want).sum())
        bad += diff
        top = lambda v: [int(i) for i in np.argsort(-v.reshape(-1))[:5]]
        print(f"  [verify] whole network (dog.jpg): {diff} / {want.size} logits "
              f"differ; top-5 CPU {top(want)} vs AIE-path {top(got)}")
    print("  [verify] PASS -- wrapper is bit-exact vs TVM" if bad == 0
          else "  [verify] FAIL")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main(*sys.argv[1:]))
