#!/usr/bin/env python3
###############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
"""Instantiate the AIEDNN template kernel gemm_kacc_bf16.cc once per layer shape.

The kernel is a C++ template, gemm_kacc<M, N, K, TK, RELU>; aiehlc instantiates
it at the launch, and every port shape and loop bound becomes a compile-time
constant there (they are the DMA windows and buffer descriptors), so one
argument set is one compiled kernel. This script does what a library generator
does: pick the template arguments for each layer, de-duplicate identical sets,
and write one source per distinct set (the GM ... GRELU instance block that the
launch `gemm_kacc<GM, GN, GK, GTK, GRELU><<<mesh>>>` reads) plus a dispatch
table (layer -> variant).

    python3 example/aiednn/gen_variants.py --out /tmp/variants            # the demo layer list
    python3 example/aiednn/gen_variants.py --out DIR --layers layers.json  # [{"m":..,"n":..,"k":..,"relu":0|1}, ...]

Build one variant like any example (absolute path; aout/ lands in the cwd):
    source script/aiehlc.sh --aie-version 5 --runtime-source-file $DIR/gemm_kacc_v0.cc
"""
import argparse
import json
import pathlib
import re
import shutil

HERE = pathlib.Path(__file__).resolve().parent
TEMPLATE = HERE / "gemm_kacc_bf16.cc"
HEADER = HERE / "gemm_kacc_bf16.h"

MESH = 4          # 4 x 4 cores
TM = TN = 16      # fixed by the kernel's 2 x 2 mmul<8,8,8> register blocking
PORT_BUDGET = 29696  # bytes of the core's port-buffer region (32 KB - 3 KB)
BF16 = 2

DEMO_LAYERS = [  # (M pixels, N channels, K = Cin*kh*kw, ReLU) -- conv layers as im2col GEMMs
    {"m": 256, "n": 64, "k": 2304, "relu": 1},
    {"m": 256, "n": 64, "k": 2304, "relu": 1},   # same shape again: re-uses variant 0
    {"m": 512, "n": 64, "k": 1152, "relu": 0},
    {"m": 64, "n": 64, "k": 576, "relu": 1},
]


def ceil_to(v, a):
    return -(-v // a) * a


def pick(layer):
    """Parameters for one layer: pad M / K, then the largest K chunk that fits."""
    m = ceil_to(layer["m"], MESH * TM)       # every core gets whole 16-row tiles
    n = layer["n"]
    if n != MESH * TN:
        raise SystemExit(f"layer N={n}: this template covers N = {MESH * TN} (one 16-wide tile per mesh col); "
                         "a wider N needs an N loop in the kernel -- another template")
    tk = 0
    for cand in (256, 128, 64, 32, 16, 8):   # largest first: fewer chunks, fewer lock round trips
        # A and W chunks double-buffered (bf16), C tile double-buffered (bf16), the fp32
        # accumulator lives in registers / the stack, not in the port region.
        mem = 2 * (TM * cand + TN * cand) * BF16 + 2 * TM * TN * BF16
        if mem <= PORT_BUDGET and cand <= layer["k"]:
            tk = cand
            break
    if not tk:
        raise SystemExit(f"layer K={layer['k']}: no K chunk fits the {PORT_BUDGET} B port region")
    k = ceil_to(layer["k"], tk)              # zero K padding is exact (0 * w = 0)
    waste = 1 - layer["m"] * layer["k"] / (m * k)
    return {"GM": m, "GN": n, "GK": k, "GTK": tk, "GRELU": int(layer["relu"])}, waste


def instantiate(params, name, out):
    src = TEMPLATE.read_text()
    for key, val in params.items():
        src, hits = re.subn(rf"^#define {key} .*$", f"#define {key} {val}", src, flags=re.M)
        if hits != 1:
            raise SystemExit(f"{TEMPLATE.name}: no unique '#define {key}' in the instance block")
    (out / f"{name}.cc").write_text(src)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", help="JSON list of {m, n, k, relu}; default: the demo list")
    a = ap.parse_args()
    layers = json.loads(pathlib.Path(a.layers).read_text()) if a.layers else DEMO_LAYERS
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy(HEADER, out / HEADER.name)   # aiehlc copies the headers next to the source
    variants, table = {}, []
    for i, layer in enumerate(layers):
        params, waste = pick(layer)
        key = tuple(sorted(params.items()))
        if key not in variants:
            variants[key] = f"gemm_kacc_v{len(variants)}"
            instantiate(params, variants[key], out)
        table.append({"layer": i, "variant": variants[key], **params, "pad_waste": round(waste, 4)})
        print(f"layer {i}: M={layer['m']} N={layer['n']} K={layer['k']} relu={layer['relu']} -> {variants[key]} "
              f"(TK {params['GTK']}, padded M {params['GM']} K {params['GK']}, waste {waste:.1%})")
    (out / "dispatch.json").write_text(json.dumps(table, indent=1) + "\n")
    print(f"{len(layers)} layers -> {len(variants)} variants in {out} (dispatch.json: layer -> variant)")


if __name__ == "__main__":
    main()
