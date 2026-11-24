###############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
"""Offload specs in every spelling a user might type -> ``[(layer_ref, source)]``.

    "layer1=aie/stem.cc"                    one pair
    "layer1=a.cc; layer4-9=b.cc"            several
    "aie/conv3x3.cc"                        every layer the kernel fits (auto)
    ("layer1", "aie/stem.cc")               a tuple
    {"layer1": "a.cc", "layer4": "b.cc"}    a dict
    [ ...any of the above... ]              a list

Launch planning for a conv offloaded as an im2col GEMM (Y[N][M] = W[N][K] . X[M][K]):

    plan_conv(m, k, n, mode="tile")     one launch per 64x64x64 tile, K summed on the host
                                        -> ceil(M/64)*ceil(N/64)*ceil(K/64) launches
    plan_conv(m, k, n, mode="kincore")  ONE launch per layer on the 4x4 mesh, K accumulated
                                        in the core (fp32, gemm_kacc walk = kincore_geom)
    plan_conv_layers(onnx, rows, mode)  the same for ONNX Conv nodes (geometry from the graph)

GAP-aiegraph-offload-7c4415: the kincore walk used to live only in
the diffusion example's host script (kacc_geom, 0c5a932 / a2e0099).
"""

from __future__ import annotations

from pathlib import Path

from .errors import AiegraphError

__all__ = ["parse_offload", "group_by_source", "PLAN_MODES", "kincore_geom", "plan_conv",
           "conv_gemm_shapes", "plan_conv_layers"]


def _one(item) -> list:
    if isinstance(item, dict):
        return [(k, v) for k, v in item.items()]
    if isinstance(item, tuple) and len(item) == 2:
        return [(item[0], item[1])]
    if isinstance(item, (str, Path)):
        out = []
        for part in str(item).split(";"):
            part = part.strip()
            if not part:
                continue
            if "=" in part:
                ref, src = part.split("=", 1)
                out.append((ref.strip(), src.strip()))
            else:
                out.append(("auto", part))
        return out
    raise AiegraphError("offload", f"cannot read offload spec {item!r}",
                        hint='use "layer1=kernel.cc", ("layer1", "kernel.cc") or '
                             '{"layer1": "kernel.cc"}')


def parse_offload(spec) -> list:
    """Normalize *spec* to ``[(layer_ref, Path)]``; sources must exist."""
    if spec is None:
        return []
    items = spec if isinstance(spec, list) else [spec]
    pairs = [p for item in items for p in _one(item)]
    for _ref, src in pairs:
        if not Path(src).is_file():
            raise AiegraphError("offload", f"kernel source {src} does not exist",
                                hint="paths are relative to the current directory")
    return [(ref, Path(src).resolve()) for ref, src in pairs]


def group_by_source(pairs, resolve) -> dict:
    """``{source: [layer indices] | None}`` -- None means "every layer it fits"."""
    out = {}
    for ref, src in pairs:
        if ref in (None, "auto", "*"):
            out[src] = None
            continue
        if out.get(src, []) is None:
            continue
        out.setdefault(src, [])
        out[src] += [i for i in resolve(ref) if i not in out[src]]
    return out


# ───────────────────────── launch planning (tile | kincore) ─────────────────────────
PLAN_MODES = ("tile", "kincore")
MESH = 4                      # 4x4 core mesh, one 64x64 output macro tile = 4 x 16 per side
TILE = 64                     # tile mode: one 64x64x64 GEMM per launch
PORT_BUFFER_BYTES = 29696     # aiehlc trace_routing kPortBufferBytes (ping-pong port windows)


def _ceil(a: int, b: int) -> int:
    return -(-a // b)


def kincore_geom(ch: int, k: int, px: int) -> dict:
    """Per-layer kincore walk of Y[ch][px] = W[ch][k] . X[px][k] (bf16 in, fp32 acc) on the 4x4 mesh.

    Same arithmetic as the example's kacc_geom (the board-proven P4bg/P4bx kernels):
      core (pm, pn) owns ch/4 channels (ntile 16-row tiles) x px/4 pixels (S 16-wide sub-tiles);
      a (W) block {16, tkw} words, b (X) block {px/4, tkw}, c (Y) block {16, px/4} fp32 = the
      accumulator, held over the whole K walk; K chunk = 2*tkw bf16, tkw = min(32, 4096/px).
      If the ping-pong windows exceed the port buffer, the GEMM runs transposed (tr=1:
      a = X, b = W, c = Y^T). K is padded by one zero chunk when ch == k/2 (square W port,
      aiehlc GemmSpace mis-sizes it).
    Returns the geometry with ``legal`` and ``why`` (empty when legal) instead of asserting."""
    why = [f"{nm}%64" for nm, v in (("ch", ch), ("k", k), ("px", px)) if v % 64]
    if why:
        return dict(legal=False, why=why, launches=None)
    tkw = min(32, 4096 // px) if px <= 4096 else 0
    tr = tkw == 0 or 2 * (16 * tkw * 4 + (px // 4) * tkw * 4 + 16 * (px // 4) * 4) > PORT_BUFFER_BYTES
    if tr:
        tkw = 32
        while tkw >= 4 and ((ch // 4) * tkw > 1024 or (k // 2) % tkw):
            tkw //= 2
    if tkw < 4 or (k // 2) % tkw:
        why.append(f"tkw{tkw}")
        return dict(legal=False, why=why, launches=None)
    kp = k + 2 * tkw if ch == k // 2 or (tr and px == k // 2) else k   # tr: a = X [px][k/2] square too
    rows, cols = (px, ch) if tr else (ch, px)
    l1 = 2 * (16 * tkw * 4 + (cols // 4) * tkw * 4 + 16 * (cols // 4) * 4)
    if l1 > PORT_BUFFER_BYTES:
        why.append(f"l1{l1}>{PORT_BUFFER_BYTES}")
    return dict(ntile=rows // 64, S=cols // 64, tkw=tkw, kch=2 * tkw, nkc=(kp // 2) // tkw, kp=kp,
                tr=int(tr), arows=rows, brows=cols, l1=l1, legal=not why, why=why,
                launches=1 if not why else None)


def plan_conv(m: int, k: int, n: int, mode: str = "kincore") -> dict:
    """Launch plan of one im2col conv GEMM (M pixels, K = C*KH*KW, N out channels).

    ``tile``: ceil(M/64)*ceil(N/64)*ceil(K/64) launches, host K-sum. ``kincore``: one launch,
    K in the core; a layer the kincore walk cannot take falls back to ``tile``
    (``mode`` in the result says which one was planned, ``why`` says why)."""
    if mode not in PLAN_MODES:
        raise AiegraphError("plan", f"mode={mode!r}", hint=f"one of {PLAN_MODES}")
    tile = _ceil(m, TILE) * _ceil(n, TILE) * _ceil(k, TILE)
    out = dict(m=m, k=k, n=n, tile_launches=tile)
    if mode == "kincore":
        g = kincore_geom(n, k, m)
        if g["legal"]:
            return dict(out, mode="kincore", launches=1, geom=g, why=[])
        return dict(out, mode="tile", launches=tile, geom=None, why=g["why"])
    return dict(out, mode="tile", launches=tile, geom=None, why=[])


def _dims(vi) -> list:
    return [d.dim_value if d.HasField("dim_value") else None for d in vi.type.tensor_type.shape.dim]


def conv_gemm_shapes(model) -> dict:
    """ONNX model (or path) -> {Conv node name: dict(m, k, n, ...)} from the graph's own shapes
    (runs onnx shape inference when an output shape is missing). Grouped/dilated convs are
    reported with ``why`` and no GEMM."""
    import onnx
    from onnx import numpy_helper  # noqa: F401  (onnx required)
    m = onnx.load(str(model)) if isinstance(model, (str, Path)) else model
    shp = {v.name: _dims(v) for v in list(m.graph.value_info) + list(m.graph.output) + list(m.graph.input)}
    shp.update({t.name: list(t.dims) for t in m.graph.initializer})
    convs = [n for n in m.graph.node if n.op_type == "Conv"]
    def _known(v):
        return v is not None and len(v) == 4 and None not in v

    if not all(_known(shp.get(n.output[0])) for n in convs):
        m = onnx.shape_inference.infer_shapes(m)
        shp.update({v.name: _dims(v) for v in list(m.graph.value_info) + list(m.graph.output)})
    out = {}
    for n in convs:
        at = {a.name: onnx.helper.get_attribute_value(a) for a in n.attribute}
        w, y = shp.get(n.input[1]), shp.get(n.output[0])
        if w is None or len(w) != 4 or not _known(y):
            out[n.name] = dict(why=["shape unknown"])
            continue
        O, CI, KH, KW = w
        why = []
        if at.get("group", 1) != 1:
            why.append(f"group{at['group']}")
        if list(at.get("dilations", [1, 1])) != [1, 1]:
            why.append("dilation")
        out[n.name] = dict(m=y[0] * y[2] * y[3], k=CI * KH * KW, n=O, weight=n.input[1],
                           kernel=[KH, KW], strides=list(at.get("strides", [1, 1])),
                           pads=list(at.get("pads", [0, 0, 0, 0])), why=why)
    return out


def plan_conv_layers(model, nodes=None, mode: str = "kincore") -> list:
    """Plan every ONNX Conv in ``nodes`` (default: all) -> one plan_conv dict per node, in order."""
    shapes = conv_gemm_shapes(model)
    rows = []
    for name in (nodes if nodes is not None else list(shapes)):
        g = shapes.get(name)
        if g is None:
            raise AiegraphError("plan", f"{name} is not a Conv of the model",
                                hint="node names as in the ONNX graph (partition.json 'node')")
        if g["why"]:
            rows.append(dict(node=name, mode="cpu", launches=None, why=g["why"]))
            continue
        rows.append(dict(plan_conv(g["m"], g["k"], g["n"], mode), node=name, weight=g["weight"]))
    return rows
