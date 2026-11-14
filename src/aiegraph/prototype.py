###############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
"""``ag.offload_prototype``: a kernel-file BACKBONE for a layer, and its contract.

    ag.offload_prototype()                    # the kernel-file format, no model needed
    proto = emod.offload_prototype("layer4")  # TVM layer 4: signature, AIEK_SPEC, backbone
    proto.write("my_kernels/")                # aiek_layer04.cc + aiek_layer04_ref.c
    # ... fill in the TODO(...)s ...
    emod.offload_check("layer4", "my_kernels/aiek_layer04.cc")

The backbone has the architecture of ``src/aietensorop/conv2dstem/conv2dstem_uk.cc``
(+ the ``conv2dstem.cc`` it includes), specialized to the layer's parameters;
what is the layer's is done, what is the algorithm's is the user's:

  1. ``<entry>_kernel`` -- the AIE kernel: ports with the layer's staged shapes,
     the mesh split, and the block loop (``aie::program_id``, ``load_buffer`` /
     ``store_buffer`` / ``release`` / ``commit``) on a tile the kernel analysis
     (``build/aiehlc-convtile trace``) accepts, so aiehlc can derive the DMA.
     TODO: the compute;
  2. ``<entry>_run`` -- the host function (``conv2d_stem_nchwc``'s role) taking
     every TVM argument in call order: staging buffers, mesh partition and the
     launch. TODO: stage the inputs / parameters, unpack the output;
  3. ``AIEK_ENTRY(<entry>)`` -- all TVM arguments into ``<entry>_run``;
  4. ``AIEK_SPEC(<entry>)`` -- proven against every call of the layer.

``<entry>_ref.c`` is the x86 reference backbone with TVM's layouts spelled out.
Until the TODOs are filled the run function returns -1 and the entry
``AIEK_ERR_KERNEL``. Kernel frames exist for TVM NCHWc conv2d layers; any other
layer gets the entry + spec skeleton.
"""

from __future__ import annotations

import os
import re
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from . import _env
from .errors import AiegraphError

__all__ = ["OffloadPrototype", "ProtoArg", "KernelFormat", "Prototypes", "layer_prototype"]

_C_TYPE = {"int8": "int8_t", "int16": "int16_t", "int32": "int32_t", "int64": "int64_t",
           "uint8": "uint8_t", "uint16": "uint16_t", "uint32": "uint32_t", "uint64": "uint64_t",
           "float16": "uint16_t", "float32": "float", "float64": "double"}

_REPO = Path(__file__).resolve().parents[2]

_FORMAT = """\
AIE user-kernel file format  (contract: include/aie_user_kernel.h,
example: src/aietensorop/conv2dstem/conv2dstem_uk.cc + conv2dstem.cc)

A kernel that replaces a network layer is ONE aiehlc source file (library mode:
no main()), in this order:

  1. the AIE kernel: the aiehlc kernel keyword on a function whose ports are
     aie::port<window, aie::IFM|WEIGHTS|OFM, aie::shape<...>>, moving data with
     aie::load_buffer / aie::store_buffer / aie::release / aie::commit (aiehlc
     derives and verifies the DMA from those calls);
  2. the host function that runs it (conv2d_stem_nchwc in the stem): stage the
     layer's tensors into the kernel's layout, partition the mesh and launch
     the kernel in the SAME function, unpack the outputs;
  3. #include "aie_user_kernel.h"  -- after the kernel;
  4. the entry, fixed signature:
         AIEK_ENTRY(my_kernel) { ... return AIEK_OK; }
     = extern "C" int my_kernel(const aiek_tensor *args, int num_args)
     args[] are the layer's tensors in TVM call order: every input as the graph
     passes it (activations `in` and constants `param`, interleaved), then every
     output `out`. Read them with AIEK_PTR(&args[i], const int8_t) etc.; call
     the host function; return AIEK_OK or AIEK_ERR_PARAM/_ALLOC/_KERNEL;
  5. the signature as data:
         AIEK_SPEC(my_kernel,
             "kernel    my_kernel\\n"
             "op        conv2d\\n"
             "match     contrib_conv2d_NCHWc\\n"      every match must be in the TVM symbol
             "reference my_kernel_ref.c\\n"           x86 reference, next file
             "in    int8  [1,1,230,230,3]  ifm\\n"     one line per argument, call order
             "param int8  [16,1,7,7,3,4]   weight\\n"
             "param int32 [1,16,1,1,4]     zero_point  require=zero\\n"
             "out   uint8 [1,16,112,112,4] output  tol=0\\n");
     dims: integer, `*`, or a NAME that must agree everywhere it appears;
     require=zero|nonzero|uniform|nonneg|range(lo,hi) checks a param's values,
     tol=N bounds |reference - TVM| on an output.

Second file, plain C for x86, the bit-exact reference of the same op:
     #include "aie_user_kernel.h"
     AIEK_REFERENCE(my_kernel) { ... }          defines my_kernel_ref()

Rules (each breaks the aiehlc frontend SILENTLY; `ag.lint(file)` checks them):
  - the aiehlc kernel keyword and the triple-angle launch token never in a
    comment or a string;
  - helpers the kernel body calls are #define macros in the .cc itself;
  - nothing that reaches the launch is declared above the kernel.
You do NOT write the TVM packed-call function: offload generates one adapter
per replaced layer that binds TVM's tensors, re-checks dtype/shape and calls
your entry.

Gates (all on x86, none on a board): lint -> detect (the spec vs every layer)
-> calibrate (reference vs TVM) -> emulate (whole network) -> build -> link.

A runnable file for one layer:   emod.offload_prototype("layer1").write("dir/")
                                 emod.offload_prototype(region="conv1")
"""


class KernelFormat(str):
    """The general file format (``ag.offload_prototype()``); prints as text."""

    def __repr__(self) -> str:
        return str(self)


@dataclass
class ProtoArg:
    index: int
    kind: str              # in | param | out
    dtype: str
    shape: tuple
    name: str
    role: str
    facts: str = ""        # value summary of a param over every call
    requires: tuple = ()   # require= keys

    @property
    def ctype(self) -> str:
        return _C_TYPE.get(self.dtype, "uint8_t")

    @property
    def var(self) -> str:
        return re.sub(r"\W", "_", self.name)

    def spec_line(self) -> str:
        shape = "[" + ",".join(str(d) for d in self.shape) + "]"
        keys = " ".join(f"require={r}" for r in self.requires)
        return f"{self.kind:<5} {self.dtype:<6} {shape:<22} {self.name}" + (f"  {keys}" if keys else "")


# ── the layer family the kernel frame covers ─────────────────────────────

@dataclass
class ConvGeom:
    """A TVM NCHWc conv2d: in [1,Cc,H,W,cb], weights [Kc,Cc,KH,KW,cb,kb], out
    [1,Kc,OH,OW,kb] (input already padded by TVM). The kernel frame works on
    HWC slabs: in [H,W,CA], weights [K,KH,KW,CA], out [OH,OW,K], CA = C rounded
    up to 4."""

    Cc: int
    cb: int
    H: int
    W: int
    Kc: int
    kb: int
    KH: int
    KW: int
    OH: int
    OW: int
    S: int
    w_slot: int
    CA: int = 0
    rows: int = 4          # mesh
    cols: int = 4
    region: str = "{3, 6, 0, 6}"   # device.partition region
    oh_t: int = 0          # on-core output tile
    ow_t: int = 0

    @property
    def C(self) -> int:
        return self.Cc * self.cb

    @property
    def K(self) -> int:
        return self.Kc * self.kb

    @property
    def taps(self) -> int:
        return self.KH * self.KW


def _conv_geom(c0):
    """(ConvGeom, None) for a TVM NCHWc conv2d call, else (None, why)."""
    if "conv2d_NCHWc" not in c0.symbol:
        return None, "not a TVM NCHWc conv2d: no kernel frame for this op yet"
    ins = [x for x in c0.args if x.kind == "in"]
    outs = [x for x in c0.args if x.kind == "out"]
    wts = [x for x in c0.args if x.kind == "param" and len(x.shape) == 6]
    if len(ins) != 1 or len(outs) != 1 or len(wts) != 1 or len(ins[0].shape) != 5 or len(outs[0].shape) != 5:
        return None, "needs one input [1,Cc,H,W,cb], one weight [Kc,Cc,KH,KW,cb,kb] and one output [1,Kc,OH,OW,kb]"
    x, w, o = ins[0], wts[0], outs[0]
    if x.dtype != "int8" or w.dtype != "int8":
        return None, f"int8 input and weights only (got {x.dtype} x {w.dtype})"
    _, Cc, H, W, cb = map(int, x.shape)
    Kc, Cc2, KH, KW, cb2, kb = map(int, w.shape)
    _, Kc2, OH, OW, kb2 = map(int, o.shape)
    if (Cc2, cb2, Kc2, kb2) != (Cc, cb, Kc, kb):
        return None, "input / weight / output channel blocking disagree"
    S = next((s for s in range(1, 5) if (H - KH) // s + 1 == OH and (W - KW) // s + 1 == OW), 0)
    if not S:
        return None, f"cannot infer the stride from {H}x{W} -> {OH}x{OW} with a {KH}x{KW} kernel"
    g = ConvGeom(Cc, cb, H, W, Kc, kb, KH, KW, OH, OW, S, w.slot)
    g.CA = (g.C + 3) // 4 * 4
    return g, None


def _divisors(n: int) -> list:
    return [d for d in range(1, n + 1) if n % d == 0]


# Meshes tried: (rows, cols, device.partition region). A 4x8 mesh crashes the
# pipeline (DmapToDmaphop) today, so only the proven 4x4 one.
_MESHES = ((4, 4, "{3, 6, 0, 6}"),)


def _tile_candidates(g: ConvGeom) -> list:
    """(rows, cols, region, oh_t, ow_t): fewest rounds first, pre-filtered by the
    budgets convautotile and the pipeline enforce -- every buffer sized at
    l1_budget (the larger of the weight / output blocks, >= 4 KB), 3 tensors,
    pp_depth deep (<= 48 KB); the 32 KB port-buffer region minus klog + statics
    for ping + pong of all three (always declared); the stack."""
    out = []
    for rows, cols, region in _MESHES:
        if g.OH % rows or g.K % cols:
            continue
        oc_t, oh_core = g.K // cols, g.OH // rows
        wgt = oc_t * g.taps * g.CA
        for oh_t in _divisors(oh_core):
            for ow_t in _divisors(g.OW):
                if g.OW // ow_t < 2:
                    continue
                ifm = ((oh_t - 1) * g.S + g.KH) * ((ow_t - 1) * g.S + g.KW) * g.CA
                ofm = oh_t * ow_t * oc_t
                l1 = max(4096, (max(wgt, ofm) + 255) // 256 * 256)
                pp = 2 if 2 * 3 * l1 <= 49152 else 1
                # kernel.cc declares ping AND pong for every window, whatever pp_depth says
                if pp * 3 * l1 <= 49152 and 2 * (ifm + wgt + ofm) + 3072 <= 32768 and \
                        ifm + wgt + ofm + 320 <= 32768 - 1024:
                    out.append((-pp, (oh_core // oh_t) * (g.OW // ow_t), ifm, rows, cols, region, oh_t, ow_t))
    return [(r, c, reg, oh_t, ow_t) for _, _, _, r, c, reg, oh_t, ow_t in sorted(out)]


# ── the prototype ────────────────────────────────────────────────────────

@dataclass
class OffloadPrototype:
    """One layer's kernel: signature, roles, spec, and generated sources."""

    layer: int
    symbol: str
    dir: str
    calls: list
    label: str
    entry: str
    op: str
    match: str
    args: list = field(default_factory=list)
    regions: list = field(default_factory=list)
    verified: str = ""
    also_fits: list = field(default_factory=list)
    geom: ConvGeom = None
    generator: str = ""    # what the kernel source is (or why only a skeleton)
    requant: bool = False  # the reference is generated in full (conv + requantize)

    # ── spec ────────────────────────────────────────────────────────────

    def spec_text(self) -> str:
        lines = [f"kernel    {self.entry}", f"op        {self.op}", f"match     {self.match}",
                 f"reference {self.entry}_ref.c"] + [a.spec_line() for a in self.args]
        return "\n".join(lines) + "\n"

    def spec(self) -> str:
        """The ``AIEK_SPEC(...)`` C text."""
        body = "\n".join(f'    "{line}\\n"' for line in self.spec_text().splitlines())
        return f"// clang-format off\nAIEK_SPEC({self.entry},\n{body});\n// clang-format on\n"

    # ── sources ─────────────────────────────────────────────────────────

    def kernel_source(self) -> str:
        """``<entry>.cc``: AIE kernel + host run function + entry + spec."""
        return _conv_kernel_cc(self) if self.geom else _skeleton_cc(self)

    def reference_source(self) -> str:
        """``<entry>_ref.c``: the op on TVM's layouts, plain C, for calibrate."""
        return _conv_ref_c(self) if self.geom else _skeleton_ref(self)

    def write(self, out_dir) -> list:
        """Write ``<entry>.cc`` and ``<entry>_ref.c`` into *out_dir*; refuse to overwrite."""
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        files = {out / f"{self.entry}.cc": self.kernel_source(),
                 out / f"{self.entry}_ref.c": self.reference_source()}
        exist = [str(p) for p in files if p.exists()]
        if exist:
            raise AiegraphError("offload_prototype", f"{', '.join(exist)} already exist(s)",
                                hint="pick another directory or entry= name")
        for p, text in files.items():
            p.write_text(text)
        print(f"[aiegraph] wrote {', '.join(str(p) for p in files)}")
        return list(files)

    def __repr__(self) -> str:
        w = max(len(a.name) for a in self.args)
        calls = ", ".join(f"[{c:02d}]" for c in self.calls)
        lines = [f"Offload prototype of layer{self.layer}: {self.label}  (calls {calls})",
                 f"  TVM symbol: {self.symbol}"]
        if self.regions:
            lines.append(f"  regions:    {', '.join(self.regions)}")
        lines += ["", f"  entry: AIEK_ENTRY({self.entry})  ->  "
                      f"int {self.entry}(const aiek_tensor *args, int num_args)",
                  f"  args[] in TVM call order ({len(self.args)}):"]
        for a in self.args:
            shape = f"{a.dtype}[{','.join(map(str, a.shape))}]"
            lines.append(f"    args[{a.index}]  {a.kind:<5} {shape:<24} {a.name:<{w}}  {a.role}"
                         + (f"  -- {a.facts}" if a.facts else ""))
        lines += ["", "  AIEK_SPEC:"] + ["    " + l for l in self.spec().splitlines()[1:-1]]
        lines += ["", f"  verified: {self.verified}"]
        if self.also_fits:
            lines.append(f"  the same spec also fits: {', '.join(f'layer{l}' for l in self.also_fits)}")
        lines += ["", f"  kernel source: {self.generator}",
                  f"  proto.write('dir/') -> {self.entry}.cc (kernel + host + entry + spec), "
                  f"{self.entry}_ref.c (x86 reference)",
                  f"  then: emod.offload_check('layer{self.layer}', 'dir/{self.entry}.cc')",
                  "  format and rules: print(ag.offload_prototype())"]
        return "\n".join(lines)


class Prototypes(list):
    """Several layers' prototypes (a multi-layer reference); prints each."""

    def __repr__(self) -> str:
        return "\n\n".join(repr(i) for i in self)


# ── generated sources: the backbone the user fills in ────────────────────

def _layout(a, layout: str) -> str:
    return f"{a.dtype}[{','.join(map(str, a.shape))}] {layout}"


def _conv_kernel_cc(p: OffloadPrototype) -> str:
    g, e = p.geom, p.entry
    oc_t = g.K // g.cols
    x = next(a for a in p.args if a.kind == "in")
    o = next(a for a in p.args if a.kind == "out")
    w = p.args[g.w_slot]
    others = [a for a in p.args if a.kind == "param" and a.index != g.w_slot]
    sig = ", ".join(f"{'' if a.kind == 'out' else 'const '}{a.ctype} *{a.var}" for a in p.args)
    call = ", ".join(a.var for a in p.args)
    ptrs = "\n".join(f"    {'' if a.kind == 'out' else 'const '}{a.ctype} *{a.var} = AIEK_PTR(&args[{a.index}], "
                     f"{'' if a.kind == 'out' else 'const '}{a.ctype}); // {a.kind} {a.dtype}"
                     f"[{','.join(map(str, a.shape))}] {a.role}" + (f" -- {a.facts}" if a.facts else "")
                     for a in p.args)
    plist = "\n".join(f" *     {a.var:<14} {a.dtype}[{','.join(map(str, a.shape))}]  {a.role}"
                      + (f" -- {a.facts}" if a.facts else "") for a in others) or " *     (none)"
    unused = "".join(f"    (void){a.var};\n" for a in others)
    return f"""\
/******************************************************************************
 * {e}.cc -- AIE user-kernel BACKBONE for layer {p.layer} ({p.label}).
 * TVM symbol: {p.symbol}
 *
 * Generated by aiegraph.offload_prototype. The structure of
 * src/aietensorop/conv2dstem/conv2dstem_uk.cc (+ conv2dstem.cc) for THIS layer:
 *   {e}_kernel   AIE core: ports, block loop and DMA contract done; compute TODO
 *   {e}_run      host: buffers + mesh launch done; staging / unpacking TODO
 *   AIEK_ENTRY     all {len(p.args)} TVM arguments -> {e}_run (done)
 *   AIEK_SPEC      the signature, proven against every call of the layer (done)
 * Every TODO(...) below is yours. Until they are filled, {e}_run returns -1 and
 * the entry AIEK_ERR_KERNEL, so nothing wrong is computed silently.
 *
 * Geometry: in [{g.H},{g.W},{g.C}] (padded by TVM), {g.KH}x{g.KW} stride {g.S} ->
 * out [{g.OH},{g.OW},{g.K}]. Mesh {g.rows}x{g.cols}: output rows over mesh rows, output
 * channels over mesh columns; each round one {g.oh_t}x{g.ow_t}x{oc_t} output tile
 * (a tile the kernel analysis accepted: the DMA is derived from the loop below).
 * Layer parameters besides the weights (they reach the core only through a port,
 * e.g. the weights' spare channels as conv2d_stem_nchwc does, or are applied on
 * the host):
{plist}
 *
 *   python: emod.offload_check("layer{p.layer}", "{e}.cc")   lint, detect, calibrate, emulate (x86)
 *           emod.offload("layer{p.layer}", "{e}.cc")         + aiehlc build + link
 ******************************************************************************/
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define AK_H {g.H}
#define AK_W {g.W}
#define AK_C {g.C}   // real input channels
#define AK_CA {g.CA}  // staged input channels (C rounded up to 4)
#define AK_KH {g.KH}
#define AK_KW {g.KW}
#define AK_S {g.S}
#define AK_OH {g.OH}
#define AK_OW {g.OW}
#define AK_K {g.K}

__global__ void {e}_kernel(aie::port<input_window_int8 *, aie::IFM, aie::shape<AK_H, AK_W, AK_CA>> x_t,
                           aie::port<input_window_int8 *, aie::WEIGHTS, aie::shape<AK_K, AK_KH, AK_KW, AK_CA>> w_t,
                           aie::port<output_window_int8 *, aie::OFM, aie::shape<AK_OH, AK_OW, AK_K>> y_t) {{
    const int oh_t = {g.oh_t}, ow_t = {g.ow_t}, oc_t = AK_K / {g.cols}; // output block per round
    const int ih_t = (oh_t - 1) * AK_S + AK_KH, iw_t = (ow_t - 1) * AK_S + AK_KW; // its input footprint
    const int oh_core = AK_OH / aie::num_programs(0);
    const int pid_h = aie::program_id(0); // mesh row: which band of output rows
    const int pid_c = aie::program_id(1); // mesh col: which output channels
    // This core's filters: w[(oc * AK_KH + kh) * AK_KW + kw) * AK_CA + c], oc < oc_t.
    const int8_t *w = aie::load_buffer(w_t, {{pid_c * oc_t, 0, 0, 0}}, {{oc_t, AK_KH, AK_KW, AK_CA}});
    for (int i = 0; i < oh_core / oh_t; i++) {{
        for (int j = 0; j < AK_OW / ow_t; j++) {{
            const int oh0 = pid_h * oh_core + i * oh_t, ow0 = j * ow_t;
            // Input slab incl. halo: x[((r * iw_t) + col) * AK_CA + c], r < ih_t, col < iw_t.
            const int8_t *x = aie::load_buffer(x_t, {{oh0 * AK_S, ow0 * AK_S, 0}}, {{ih_t, iw_t, AK_CA}});
            // Output tile: y[(oh * ow_t + ow) * oc_t + oc].
            int8_t *y = aie::store_buffer(y_t, {{oh0, ow0, pid_c * oc_t}}, {{oh_t, ow_t, oc_t}});
            // TODO(compute): for oh < oh_t, ow < ow_t, oc < oc_t:
            //   acc = sum over kh < AK_KH, kw < AK_KW, c < AK_C of
            //         x[((oh * AK_S + kh) * iw_t + ow * AK_S + kw) * AK_CA + c] * w[((oc * AK_KH + kh) * AK_KW + kw) * AK_CA + c]
            //   y[(oh * ow_t + ow) * oc_t + oc] = the layer's epilogue of acc (its parameters, see the header)
            (void)x;
            (void)y;
            aie::release(x_t);
            aie::commit(y_t);
        }}
    }}
    (void)w;
    aie::release(w_t);
}}

// ═══════════════════════════════════════════════════════════════════════════
// HOST -- after the kernel: nothing that reaches the launch is declared above it.
// ═══════════════════════════════════════════════════════════════════════════
#include "aie_user_kernel.h"

static int8_t *g_{e}_x = nullptr; // [AK_H][AK_W][AK_CA]            the kernel's input
static int8_t *g_{e}_w = nullptr; // [AK_K][AK_KH][AK_KW][AK_CA]   the kernel's weights
static int8_t *g_{e}_y = nullptr; // [AK_OH][AK_OW][AK_K]          the kernel's output

static int {e}_alloc(void) {{
    if (g_{e}_x)
        return 0;
    g_{e}_x = (int8_t *)__Runtime_Alloc(AK_H * AK_W * AK_CA);
    g_{e}_w = (int8_t *)__Runtime_Alloc(AK_K * AK_KH * AK_KW * AK_CA);
    g_{e}_y = (int8_t *)__Runtime_Alloc(AK_OH * AK_OW * AK_K);
    return g_{e}_x && g_{e}_w && g_{e}_y ? 0 : -1;
}}

// Layer {p.layer} on the AIE (conv2d_stem_nchwc's role); TVM's arguments in call order.
int {e}_run({sig}) {{
    if ({e}_alloc() != 0)
        return -1;
    // TODO(stage input): {x.var} {_layout(x, f"(NCHW{g.cb}c)")} -> g_{e}_x [AK_H][AK_W][AK_CA] (HWC; channels >= AK_C zero)
    // TODO(stage weights): {w.var} {_layout(w, f"(OIHW{g.cb}i{g.kb}o)")} -> g_{e}_w [AK_K][AK_KH][AK_KW][AK_CA]
    // TODO(parameters): get the other parameters (header) to the kernel or apply them below
    aieSetDevice(0);
    aieArray device;
    aieMesh mesh = device.partition({g.region}, {g.rows}, {g.cols});
    {e}_kernel<<<mesh>>>(g_{e}_x, g_{e}_w, g_{e}_y, {g.OH * g.OW}, {g.K}, {g.taps * g.CA});
    // TODO(unpack output): g_{e}_y [AK_OH][AK_OW][AK_K] (HWC) -> {o.var} {_layout(o, f"(NCHW{g.kb}c)")}
    (void){x.var};
    (void){w.var};
{unused}    (void){o.var};
    return -1; // TODO: return 0 once the kernel computes and the staging above is done
}}

AIEK_ENTRY({e}) {{
    (void)num_args; // the generated adapter already checked count, dtypes and shapes
{ptrs}
    return {e}_run({call}) == 0 ? AIEK_OK : AIEK_ERR_KERNEL;
}}

{p.spec()}"""


# The fused conv + requantize signature TVM emits for these int8 networks
# (zp_a, bias, zp_b, multiplier, unused, shift around the weights): the
# reference for it is generated in full, bit-exact (calibrated on ResNet-18).
_REQUANT_ROLES = {2: "zero-point", 3: "bias", 4: "zero-point", 5: "requant multiplier", 7: "requant shift"}


def _is_requant(p: OffloadPrototype, roles: dict) -> bool:
    a, g = p.args, p.geom
    if g is None or len(a) != 9 or g.w_slot != 1 or a[8].dtype not in ("int8", "uint8"):
        return False
    vec, per = (1, g.Kc, 1, 1, g.kb), (g.Kc, g.kb)
    shapes = {2: vec, 3: vec, 4: vec, 5: per, 6: per, 7: per}
    if any(a[i].dtype != "int32" or a[i].shape != sh for i, sh in shapes.items()):
        return False
    if any(roles.get(i) != r for i, r in _REQUANT_ROLES.items()):
        return False
    return a[6].requires == ("zero",)


def _requant_ref_c(p: OffloadPrototype) -> str:
    """The full reference: int32 conv on TVM's layouts + TVM's requantize."""
    g, e, a = p.geom, p.entry, p.args
    ctype = _C_TYPE[a[8].dtype]
    clamp = "v < 0 ? 0 : v > 255 ? 255 : v"
    out = f"({ctype})({clamp})" if a[8].dtype == "uint8" else f"({ctype})(({clamp}) - 128)"
    ptrs = "\n".join(f"    {'' if x.kind == 'out' else 'const '}{x.ctype} *{x.var} = AIEK_PTR(&args[{x.index}], "
                     f"{'' if x.kind == 'out' else 'const '}{x.ctype}); // {x.kind} {x.dtype}"
                     f"[{','.join(map(str, x.shape))}] {x.role}" for x in a)
    return f"""\
/******************************************************************************
 * {e}_ref.c -- x86 reference of layer {p.layer} ({p.label}), plain C on TVM's own
 * layouts; offload_check calibrates it against TVM bit for bit and swaps it into
 * the whole network. Generated by aiegraph.offload_prototype -- COMPLETE: the
 * layer's math, the spec of what the AIE kernel ({e}.cc) must compute.
 *   acc = sum x[cc,oh*{g.S}+kh,ow*{g.S}+kw,cb] * w[kc,cc,kh,kw,cb,kb]                 int32
 *   v   = (int32)(((int64)(acc + bias - zp_a - zp_b) * mult + 2^(sh+30)) >> (sh+31))
 *   out = {"clamp(v, 0, 255)" if a[8].dtype == "uint8" else "clamp(v, 0, 255) - 128"}
 *   x  [1,{g.Cc},{g.H},{g.W},{g.cb}] NCHW{g.cb}c (padded by TVM), w [{g.Kc},{g.Cc},{g.KH},{g.KW},{g.cb},{g.kb}] OIHW{g.cb}i{g.kb}o,
 *   out [1,{g.Kc},{g.OH},{g.OW},{g.kb}] NCHW{g.kb}c
 ******************************************************************************/
#include "aie_user_kernel.h"

AIEK_REFERENCE({e}) {{
    (void)num_args;
{ptrs}
    (void){a[6].var}; // all zero (require=zero in the spec)
    for (int kc = 0; kc < {g.Kc}; kc++)
        for (int oh = 0; oh < {g.OH}; oh++)
            for (int ow = 0; ow < {g.OW}; ow++)
                for (int kb = 0; kb < {g.kb}; kb++) {{
                    int32_t acc = 0;
                    for (int cc = 0; cc < {g.Cc}; cc++)
                        for (int kh = 0; kh < {g.KH}; kh++)
                            for (int kw = 0; kw < {g.KW}; kw++)
                                for (int cb = 0; cb < {g.cb}; cb++)
                                    acc += (int32_t){a[0].var}[((cc * {g.H} + oh * {g.S} + kh) * {g.W} + ow * {g.S} + kw) * {g.cb} + cb] *
                                           (int32_t){a[1].var}[((((kc * {g.Cc} + cc) * {g.KH} + kh) * {g.KW} + kw) * {g.cb} + cb) * {g.kb} + kb];
                    const int oc = kc * {g.kb} + kb, sh = {a[7].var}[oc];
                    const int32_t t = acc + {a[3].var}[oc] - {a[2].var}[oc] - {a[4].var}[oc];
                    const int32_t v = (int32_t)(((int64_t)t * (int64_t){a[5].var}[oc] + ((int64_t)1 << (sh + 30))) >> (sh + 31));
                    {a[8].var}[((kc * {g.OH} + oh) * {g.OW} + ow) * {g.kb} + kb] = {out};
                }}
    return AIEK_OK;
}}
"""


def _conv_ref_c(p: OffloadPrototype) -> str:
    if p.requant:
        return _requant_ref_c(p)
    g, e = p.geom, p.entry
    ptrs = "\n".join(f"    {'' if a.kind == 'out' else 'const '}{a.ctype} *{a.var} = AIEK_PTR(&args[{a.index}], "
                     f"{'' if a.kind == 'out' else 'const '}{a.ctype}); // {a.kind} {a.dtype}"
                     f"[{','.join(map(str, a.shape))}] {a.role}" for a in p.args)
    voids = "".join(f"    (void){a.var};\n" for a in p.args)
    return f"""\
/******************************************************************************
 * {e}_ref.c -- x86 reference BACKBONE of layer {p.layer} ({p.label}), plain C on
 * TVM's own layouts. offload_check calibrates it against TVM bit for bit, then
 * swaps it into the whole network. Generated by aiegraph.offload_prototype.
 *   x  [1,{g.Cc},{g.H},{g.W},{g.cb}]  NCHW{g.cb}c     x[((cc*{g.H} + h)*{g.W} + w)*{g.cb} + cb]
 *   w  [{g.Kc},{g.Cc},{g.KH},{g.KW},{g.cb},{g.kb}] OIHW{g.cb}i{g.kb}o  w[((((kc*{g.Cc} + cc)*{g.KH} + kh)*{g.KW} + kw)*{g.cb} + cb)*{g.kb} + kb]
 *   out [1,{g.Kc},{g.OH},{g.OW},{g.kb}] NCHW{g.kb}c   out[((kc*{g.OH} + oh)*{g.OW} + ow)*{g.kb} + kb]
 *   stride {g.S}; the input is already padded.
 ******************************************************************************/
#include "aie_user_kernel.h"

AIEK_REFERENCE({e}) {{
    (void)num_args;
{ptrs}
{voids}    // TODO: the layer, element by element, bit-exact with TVM (its C: layers/{p.dir}/).
    return AIEK_ERR_KERNEL;
}}
"""


def _skeleton_cc(p: OffloadPrototype) -> str:
    ptrs = "\n".join(f"    {'' if x.kind == 'out' else 'const '}{x.ctype} *{x.var} = AIEK_PTR(&args[{x.index}], "
                     f"{'' if x.kind == 'out' else 'const '}{x.ctype}); // {x.kind} {x.dtype}"
                     f"[{','.join(map(str, x.shape))}] {x.role}" for x in p.args)
    names = "; (void)".join(x.var for x in p.args)
    return f"""\
// {p.entry}.cc -- AIE user kernel for layer {p.layer} ({p.label})
// TVM symbol: {p.symbol}
// Skeleton from aiegraph.offload_prototype: {p.generator}
// Architecture to follow: src/aietensorop/conv2dstem/conv2dstem_uk.cc + conv2dstem.cc --
// the AIE kernel and a host function that stages args[], partitions the mesh and
// launches the kernel go ABOVE the include below; the entry calls that function.

#include "aie_user_kernel.h"

AIEK_ENTRY({p.entry}) {{
    (void)num_args; // the generated adapter already checked count, dtypes and shapes
{ptrs}
    (void){names};
    return AIEK_ERR_KERNEL; // until implemented
}}

{p.spec()}"""


def _skeleton_ref(p: OffloadPrototype) -> str:
    return f"""\
// {p.entry}_ref.c -- x86 reference of layer {p.layer} ({p.label}), plain C.
#include "aie_user_kernel.h"

AIEK_REFERENCE({p.entry}) {{
    (void)args;
    (void)num_args;
    return AIEK_ERR_KERNEL; // TODO: the op, element by element, bit-exact with TVM
}}
"""


# ── building one ─────────────────────────────────────────────────────────

_ROLE_NAME = {"weight": "weight", "bias": "bias", "zero-point": "zero_point",
              "requant multiplier": "requant_mult", "requant shift": "requant_shift",
              "unused": "unused", "param": "param"}


def _roles(net, layer: int, symbol: str) -> dict:
    """{slot: role} for the constants, from the layer's generated C (network_md)."""
    try:
        from frontend.tvmrelay.network_md import kernel_param_usage, param_role
        src = net.layer_source(layer)
        if src is None:
            return {}
        usage = kernel_param_usage(src.read_text(errors="replace"), symbol)
        return {slot: param_role(u) for slot, u in usage.items()}
    except Exception:  # roles are a courtesy; the spec does not depend on them
        return {}


def _facts(net, calls, slot) -> tuple:
    """(summary, suggested requires) for a param over every call of the layer."""
    import numpy as np

    lo = hi = None
    zero = uniform = True
    for c in calls:
        a = c.args[slot]
        v = net.param_array(a.param, a.dtype, a.shape).astype(np.int64 if a.dtype[0] in "iu" else np.float64)
        if v.size == 0:
            continue
        lo = v.min() if lo is None else min(lo, v.min())
        hi = v.max() if hi is None else max(hi, v.max())
        zero &= not v.any()
        uniform &= bool((v == v.flat[0]).all())
    if lo is None:
        return "", ()
    if zero:
        return "all zero", ("zero",)
    fmt = (lambda x: f"{int(x)}") if float(lo).is_integer() and float(hi).is_integer() else (lambda x: f"{x:g}")
    if uniform:
        return f"uniform {fmt(lo)}" + (" (per call)" if len(calls) > 1 else ""), ()
    return f"values in [{fmt(lo)}, {fmt(hi)}]", ()


def _op(symbol: str) -> str:
    for key in ("conv2d", "dense", "matmul", "pool", "add", "softmax"):
        if key in symbol:
            return key
    return "op"


def _match(symbol: str) -> str:
    """The symbol minus TVM's prefix and content hash: shared by every layer of
    the same fused pattern, so one kernel can serve all of them."""
    s = re.sub(r"^tvmgen_default_fused_", "", symbol)
    s = re.sub(r"_[0-9a-f]{16}_*(\d+)?_*$", "", s)
    return re.sub(r"_\d+$", "", s) or symbol


def _verify(p: OffloadPrototype, net) -> None:
    """Parse the spec with the real grammar and match it against every call."""
    from frontend.tvmrelay.aiekernel.match import detect_layers, match_call
    from frontend.tvmrelay.aiekernel.spec import load_source, parse_specs
    with tempfile.TemporaryDirectory() as tmp:
        f = Path(tmp) / f"{p.entry}.cc"
        f.write_text('#include "aie_user_kernel.h"\n' + p.spec())
        (Path(tmp) / f"{p.entry}_ref.c").write_text("")
        spec = parse_specs(load_source(f))[0]
        calls = [c for c in net.calls if c.call in p.calls]
        bad = [v for v in (match_call(spec, c, net) for c in calls) if not v.ok]
        if bad:
            raise AiegraphError("offload_prototype",
                                f"layer{p.layer}: the generated spec does not match call "
                                f"[{bad[0].call.call:02d}]: {bad[0].reasons[0]}",
                                hint="report this: the prototype generator and aiekernel disagree")
        p.verified = (f"AIEK_SPEC parses and matches {len(calls)}/{len(calls)} call(s) of "
                      f"layer{p.layer} (aiekernel.match, value checks included)")
        p.also_fits = [v.layer for v in detect_layers(spec, net)
                       if v.layer != p.layer and all(c.ok for c in v.calls)]


def _convtile() -> Path:
    exe = _REPO / "build" / "aiehlc-convtile"
    return exe if os.access(exe, os.X_OK) else None


def _pick_tile(p: OffloadPrototype) -> str:
    """Largest tile the kernel analysis accepts; returns how it was decided."""
    cands = _tile_candidates(p.geom)
    if not cands:
        g = p.geom
        return (f"no mesh ({', '.join(f'{r}x{c}' for r, c, _ in _MESHES)}) / tile fits: the weight block per core "
                f"({g.K // _MESHES[-1][1]} ch x {g.taps} taps x {g.CA} B) or the halo slab exceeds core memory "
                f"(output-channel time tiling is not in the conv path yet)")
    exe = _convtile()
    if exe is None:
        p.geom.rows, p.geom.cols, p.geom.region, p.geom.oh_t, p.geom.ow_t = cands[0]
        return "tile by the core-memory estimate (build/aiehlc-convtile missing: not traced)"
    errors = []
    for rows, cols, region, oh_t, ow_t in cands[:8]:
        p.geom.rows, p.geom.cols, p.geom.region, p.geom.oh_t, p.geom.ow_t = rows, cols, region, oh_t, ow_t
        with tempfile.TemporaryDirectory() as tmp:
            f = Path(tmp) / f"{p.entry}.cc"
            f.write_text(_conv_kernel_cc(p))
            r = subprocess.run([str(exe), "trace", str(f)], capture_output=True, text=True, timeout=900)
        if r.returncode == 0:
            return (f"mesh {rows}x{cols}, tile {oh_t}x{ow_t}x{p.geom.K // cols} accepted by the kernel analysis "
                    f"(aiehlc-convtile trace: DMA derived and every block checked)")
        errors.append(next((l for l in r.stdout.splitlines() if "ERROR" in l), r.stdout[-200:]).strip())
    p.geom.oh_t = p.geom.ow_t = 0
    return f"no tile accepted by the kernel analysis: {errors[0] if errors else '?'}"


def layer_prototype(net, layer: int, entry: str = None, regions=()) -> OffloadPrototype:
    """The prototype of manifest layer *layer* of *net* (an ``aiekernel.graph.Network``)."""
    calls = net.layers().get(layer)
    if not calls:
        raise AiegraphError("offload_prototype", f"layer{layer} has no kernel call")
    c0 = calls[0]
    for c in calls[1:]:
        if [(a.kind, a.dtype, tuple(a.shape)) for a in c.args] != \
                [(a.kind, a.dtype, tuple(a.shape)) for a in c0.args]:
            raise AiegraphError("offload_prototype", f"layer{layer}: calls disagree on the signature")
    roles = _roles(net, layer, c0.symbol)
    args, seen = [], {}
    n_in = sum(1 for a in c0.args if a.kind == "in")
    n_out = sum(1 for a in c0.args if a.kind == "out")
    k_in = k_out = 0
    for a in c0.args:
        facts, req = "", ()
        if a.kind == "param":
            role = roles.get(a.slot, "param")
            facts, req = _facts(net, calls, a.slot)
            name = _ROLE_NAME.get(role, "param")
            if name == "param":
                name = f"p{a.slot}"
        elif a.kind == "in":
            k_in += 1
            role = "activation" if k_in == 1 else "activation (residual / second input)"
            name = "ifm" if n_in == 1 else f"ifm{k_in - 1}"
        else:
            k_out += 1
            role = "output"
            name = "ofm" if n_out == 1 else f"ofm{k_out - 1}"
        seen[name] = seen.get(name, 0) + 1
        if seen[name] > 1:
            name = f"{name}{seen[name]}"
        args.append(ProtoArg(a.slot, a.kind, a.dtype, tuple(int(d) for d in a.shape),
                             name, role, facts, req))
    from .embedded import _label
    p = OffloadPrototype(layer=layer, symbol=c0.symbol, dir=c0.dir, calls=[c.call for c in calls],
                         label=_label(c0), entry=entry or f"aiek_layer{layer:02d}", op=_op(c0.symbol),
                         match=_match(c0.symbol), args=args, regions=list(regions))
    geom, why = _conv_geom(c0)
    if geom is not None:
        p.geom = geom
        how = _pick_tile(p)
        if p.geom.oh_t:
            p.generator = f"kernel + host + entry backbone ({how}); compute, staging and parameters are yours"
        else:
            p.geom, why = None, how
    if p.geom is None:
        p.generator = f"entry + spec skeleton only -- {why}"
    elif _is_requant(p, roles):
        p.requant = True
        # the reference's shift arithmetic needs sh + 30 >= 0 and sh + 31 < 64
        p.args[7].requires = p.args[7].requires + ("range(-30,32)",)
        p.generator += "; the reference _ref.c is generated complete (bit-exact conv + requantize)"
    _verify(p, net)
    return p


def kernel_format() -> KernelFormat:
    return KernelFormat(_FORMAT)


def prototype_for(emod, layer=None, region=None, entry: str = None):
    """``emod.offload_prototype``: one layer -> OffloadPrototype, several -> Prototypes."""
    if layer is None and region is None:
        return kernel_format()
    _env.frontend("")
    net = emod._network()
    if region is not None:
        regs = emod.regions.find(region)
        layers = sorted({r.layer for r in regs if r.layer is not None})
        if not layers:
            raise AiegraphError("offload_prototype", f"region {region!r} is not mapped to a TVM layer")
    else:
        layers = emod.resolve(layer)
    by_layer = {}
    if region is not None or emod.model is not None:
        try:
            for r in emod.regions:
                by_layer.setdefault(r.layer, []).append(r.label)
        except Exception:
            pass
    out = Prototypes(layer_prototype(net, l, entry if len(layers) == 1 else None, by_layer.get(l, ()))
                     for l in layers)
    return out[0] if len(out) == 1 else out
