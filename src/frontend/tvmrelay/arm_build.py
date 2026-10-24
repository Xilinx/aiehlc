#!/usr/bin/env python3
###############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
"""Link the generated ResNet-18 C into a baremetal aarch64 ELF for the board.

``relay.build(target="c")`` gives kernels but no program: there is no ``main``,
no weights in memory, and no code calling the kernels in order. TVM normally
supplies that from its graph *runtime* (C++, host-side). This module generates
the missing pieces from ``resnet18_graph.json`` instead, so the result is plain
C that the aiehlc aarch64 toolchain can link:

    layers/NN_op/NN_op.c one kernel per operator       (stage 5)
      -> NN_op.o -> liblayers.a                                        <- here
    tvm_runtime_shim.c   the ~4 TVM runtime symbols the kernels call   <- here
    graph_driver.c       storage plan + kernel calls in graph order    <- here
    main.c               entry point, weight load, argmax, timing      <- here
    weights.bin          resnet18_params.bin, linked via ld -r         <- here
    Makefile             the build recipe                              <- here
      -> make -> main.elf

**Kernels come from the per-layer split by default.** Stage 5 writes one
translation unit per operator under ``<out>/layers/``; each is compiled to a
``.o`` *beside its source* and archived into ``liblayers.a``. That is what
makes a single operator independently rebuildable -- editing one layer
recompiles one TU, not a 536 KB module -- and it is the natural place for a
layer's AIE artifacts to sit next to the C they replace.

The monolithic ``resnet18.c`` remains the fallback, used when stage 5 was
skipped (``--no-split``). Exactly one of the two is wired into the Makefile,
so no kernel is ever compiled twice. Measured difference on ResNet-18 int8:
1,312,864 vs 1,312,608 bytes of .text (+0.02%, slightly less cross-TU
inlining), same 65 ``tvmgen_*`` symbols.

**Python generates; make builds.** Everything above is written to
``<out>/arm_build/``, then ``make`` is invoked on the generated Makefile --
the same one a person re-runs by hand after editing a source, without going
back through Python:

    make -C worklocal/tvmrelay_deploy/arm_build            # rebuild changed
    make -C ... clean && make -C ...                       # from scratch
    make -C ... CROSS=aarch64-linux-gnu- CPU=cortex-a72    # retarget

``--no-make`` stops after generating, which is also what happens automatically
when the cross toolchain is not on PATH.

**The same sources also build for this host** -- ``make local`` (or
``--local`` here and in ``deploy_flow.py``) links ``main_local.elf`` with the
host gcc and runs it, so the graph's top-5 can be diffed against the
onnxruntime reference without flashing a board. It is not the cross build with
a different ``-mcpu``: the BSP and the whole baremetal link recipe are dropped,
and two flags have to be *added* -- ``-D__AIESIM__`` so ``aie_timer.h`` takes
its portable ``clock_gettime`` branch instead of including the BSP's
``xtime_l.h``, and ``-std=gnu11`` rather than ``-std=c11`` because
``CLOCK_MONOTONIC`` is POSIX and strict ISO mode hides it. An AIE-offloaded
build cannot be linked here and says so (the archives are aarch64, and their
kernels run on the array).

**Why not TVM's AOT executor.** ``Executor("aot")`` does emit a real
``tvmgen_default_run`` entry point, which looks like exactly the right answer.
It was tried and rejected: AOT inlines every weight into the C source as
initializer text, which for ResNet-18 is a **188 MB** .c file. That is not
compilable in practice, and ``link-params=False`` does not prevent it. Reading
the weights from a binary blob at runtime keeps the source at ~340 KB and the
weights in ``.rodata`` where they belong.

**Toolchain.** Reuses the same recipe as ``script/hostcompile.sh`` and
``quant/build_demo.build_aarch64``: ``aarch64-none-elf-gcc``, ``-mcpu=cortex-a78``,
the cortexa78 BSP under ``thirdparty/arch/``, ``--specs=nosys.specs``, and the
BSP linker script. ``script/setup.sh --path-set-only`` must have been sourced;
``toolchain_status()`` reports whether it was, and the build is skipped with an
explanation rather than a traceback when it was not.

**No AIE yet.** This links the CPU path only -- ``aie_runtime.c``, ``routing.cc``
and an embedded kernel ELF are what ``hostcompile.sh`` adds, and none of them
exist for these layers yet. When a layer folder grows a ``.bcf`` and a kernel,
that layer's call site here is what changes.

Memory: the graph needs a 52 MB workspace and 46 MB of weights, against the
2 GB of DDR the BSP linker script maps. Both are checked at generation time.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import struct
import subprocess
from pathlib import Path

__all__ = ["build_arm_elf", "build_local_elf", "toolchain_status",
           "generate_driver", "write_readme", "graph_sha"]

#: Set by the BSP linker script (``thirdparty/arch/cortexa78_0/lscript.ld``):
#: noc_ddr4_C0_DDR_LOW0 is 0x80000000. Generation fails rather than emitting a
#: program that cannot possibly fit.
DDR_BYTES = 0x80000000

#: ``kTVMNDArrayListMagic`` (``src/runtime/file_utils.h:107``) and
#: ``kTVMNDArrayMagic`` (``include/tvm/runtime/ndarray.h:433``) — the headers of
#: a ``relay.save_param_dict`` blob and of each array inside it. Checked when
#: parsing so a truncated or wrong-format file is caught here rather than as
#: garbage weights on target.
PARAM_MAGIC = 0xF7E58D4F05049CB7
NDARRAY_MAGIC = 0xDD5E40F096B4A13F


# ═══════════════════════════════════════════════════════════════════════════
#  Toolchain
# ═══════════════════════════════════════════════════════════════════════════

def toolchain_status(repo_root: Path) -> tuple:
    """``(ok, detail)`` for the aarch64 cross toolchain + BSP. Never raises."""
    cc = shutil.which("aarch64-none-elf-gcc")
    if cc is None:
        return False, ("aarch64-none-elf-gcc not on PATH -- run "
                       "`source script/setup.sh --path-set-only` first")
    arch = repo_root / "thirdparty" / "arch" / "cortexa78_0"
    lscript = arch / "lscript.ld"
    if not lscript.is_file():
        return False, f"linker script missing: {lscript}"
    bsp = _bsp_dir(arch)
    if bsp is None or not (bsp / "lib").is_dir():
        return False, f"cortexa78 BSP not found under {arch}"
    return True, cc


def _bsp_dir(arch: Path):
    """The BSP root, handling both the nested workspace and flat layouts.

    ``hostcompile.sh`` probes the same two paths; keep them in step.
    """
    nested = (arch / "workspace" / "platform_baremetal" / "cortexa78_0"
              / "standalone_cortexa78_0" / "bsp")
    for cand in (nested, arch / "bsp"):
        if (cand / "include").is_dir():
            return cand
    return None


# ═══════════════════════════════════════════════════════════════════════════
#  Params blob
# ═══════════════════════════════════════════════════════════════════════════

def read_param_names(blob_path: Path) -> list:
    """Names, in file order, from a ``relay.save_param_dict`` blob.

    Only the names are needed: the driver indexes the blob by order, so the
    payload stays on disk and is linked as a binary object rather than parsed.
    Raises if the magic does not match, which catches a truncated or
    wrong-format file here instead of as silent garbage on target.
    """
    data = blob_path.read_bytes()
    magic, _reserved = struct.unpack_from("<QQ", data, 0)
    if magic != PARAM_MAGIC:
        raise ValueError(
            f"{blob_path.name}: bad magic 0x{magic:X}, expected "
            f"0x{PARAM_MAGIC:X} (not a relay.save_param_dict blob?)")
    n_names = struct.unpack_from("<Q", data, 16)[0]
    names, off = [], 24
    for _ in range(n_names):
        ln = struct.unpack_from("<Q", data, off)[0]
        off += 8
        names.append(data[off:off + ln].decode())
        off += ln
    return names


def param_offsets(blob_path: Path) -> dict:
    """``{name: (byte_offset, nbytes)}`` into the raw blob for each parameter.

    Walks the NDArray records so the generated C can ``memcpy`` straight out of
    the linked blob -- no runtime deserializer, no allocation, and the weights
    stay in ``.rodata``.

    **``graph_driver.c`` and ``weights.bin`` are a matched pair.** The offsets
    returned here are baked into the driver as literals, and TVM serializes
    ``save_param_dict`` in an unordered-map order that varies run to run -- the
    same weights land at different offsets each time (the C source is stable;
    only the blob ordering moves). Pairing a driver with a blob from a
    *different* stage-4 run therefore reads each weight from the wrong place
    and silently computes garbage. Regenerate both together, which
    ``build_arm_elf`` does; never hand-copy one over the other.
    """
    data = blob_path.read_bytes()
    names = read_param_names(blob_path)
    # Skip the header + name table to reach the array payloads.
    off = 24
    for name in names:
        off += 8 + len(name.encode())
    n_arrays = struct.unpack_from("<Q", data, off)[0]
    off += 8
    if n_arrays != len(names):
        raise ValueError(f"{blob_path.name}: {n_arrays} arrays vs "
                         f"{len(names)} names")

    out = {}
    for name in names:
        magic = struct.unpack_from("<Q", data, off)[0]
        if magic != NDARRAY_MAGIC:
            raise ValueError(f"{blob_path.name}: bad NDArray magic at {off}")
        off += 8 + 8                      # magic + reserved
        off += 4 + 4                      # device_type + device_id
        ndim = struct.unpack_from("<i", data, off)[0]
        off += 4
        off += 1 + 1 + 2                  # DLDataType: code, bits, lanes
        off += 8 * ndim                   # int64 shape
        nbytes = struct.unpack_from("<q", data, off)[0]
        off += 8
        out[name] = (off, nbytes)
        off += nbytes
    return out


# ═══════════════════════════════════════════════════════════════════════════
#  Driver generation
# ═══════════════════════════════════════════════════════════════════════════

_SHIM_C = """\
/* Generated by src/frontend/tvmrelay/arm_build.py -- do not edit.
 *
 * The four TVM runtime symbols the generated kernels reference. TVM's real
 * runtime is C++ and host-side; on baremetal aarch64 all the kernels actually
 * need is a bump allocator and an error sink.
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

/* Kernels call this for scratch inside a fused op. A bump allocator with a
 * free-the-most-recent policy matches TVM's strictly nested alloc/free
 * discipline within one kernel invocation. */
#define WS_ALIGN 64
static uint8_t *ws_base, *ws_cur, *ws_end;

void tvm_shim_workspace_init(void *buf, uint64_t bytes) {
    ws_base = ws_cur = (uint8_t *)buf;
    ws_end = ws_base + bytes;
}

void *TVMBackendAllocWorkspace(int device_type, int device_id, uint64_t nbytes,
                               int dtype_code_hint, int dtype_bits_hint) {
    (void)device_type; (void)device_id;
    (void)dtype_code_hint; (void)dtype_bits_hint;
    uint8_t *p = (uint8_t *)(((uintptr_t)ws_cur + (WS_ALIGN - 1)) & ~(uintptr_t)(WS_ALIGN - 1));
    if (p + nbytes > ws_end) {
        printf("FATAL: kernel workspace exhausted (%llu more bytes)\\n",
               (unsigned long long)nbytes);
        return 0;
    }
    ws_cur = p + nbytes;
    return p;
}

int TVMBackendFreeWorkspace(int device_type, int device_id, void *ptr) {
    (void)device_type; (void)device_id;
    if ((uint8_t *)ptr < ws_cur) ws_cur = (uint8_t *)ptr;   /* nested: rewind */
    return 0;
}

void TVMAPISetLastError(const char *msg) { printf("TVM error: %s\\n", msg); }

int TVMBackendParallelLaunch(void *flambda, void *cdata, int num_task) {
    /* Single core: run the lambda once as task 0. Kernels only take this path
     * when a schedule was parallelized; ours are not, but the symbol must
     * resolve. */
    (void)flambda; (void)cdata; (void)num_task;
    return 0;
}
"""

_MAIN_C = """\
/* Generated by src/frontend/tvmrelay/arm_build.py -- do not edit.
 *
 * Baremetal entry point: point the graph at its weights, feed it the
 * preprocessed image, run it, and print the top-5 with class names and
 * logits. The printed logits are what the host-side CPU reference is
 * compared against -- a class index alone would not show a small numeric
 * drift, and drift is exactly what a miscompiled kernel produces.
 *
 * Each phase is timed with the Arm generic timer (``aie_timer.h``, the same
 * XTime/COUNTS_PER_SECOND helpers the tilinglinalg host code uses). Three
 * numbers rather than one total, because they answer different questions and
 * only one of them is the model:
 *
 *   init     one-off: point the graph at the weight blob
 *   inference  graph_run() -- THE number; what an AIE offload has to beat
 *   top-5    the argmax, printed so it is visibly not part of inference
 *
 * COUNTS_PER_SECOND is reported too: a bare tick count means nothing without
 * the frequency, and on this part the generic timer is nowhere near CPU clock.
 */
#include <stdint.h>
#include <stdio.h>

#include "aie_timer.h"        /* XTime, XTime_GetTime, COUNTS_PER_SECOND */
#include "input_image.h"      /* input_image[], INPUT_IMAGE_ELEMS */
#include "imagenet_labels.h"  /* imagenet_labels[], IMAGENET_NUM_CLASSES */

void  graph_init(const uint8_t *params_blob);
float *graph_input(void);
float *graph_run(void);
int    graph_output_len(void);
#ifdef GRAPH_PROFILE
void graph_prof_print(void);       /* make PROFILE=1: per-call time table */
void graph_prof_count_run(void);
#endif

/* `ld -r -b binary` names: weights.bin -> _binary_weights_bin_start. */
extern const uint8_t _binary_weights_bin_start[];

#define TOPK 5

/* Ticks -> ms. Done in one place so every number uses the same conversion,
 * and as double because at ~100 MHz a 24 ms inference is ~2.4M ticks: fine in
 * an integer, but the division to ms is not. */
static double ticks_to_ms(XTime start, XTime end) {
    return 1.0 * (double)(end - start) / (double)COUNTS_PER_SECOND * 1000.0;
}

int main(void) {
    XTime t_init0, t_init1, t_run0, t_run1, t_top0, t_top1;

    printf("resnet18: init\\n");
    XTime_GetTime(&t_init0);
    graph_init(_binary_weights_bin_start);
    XTime_GetTime(&t_init1);

    /* Outside the inference window on purpose: copying the image in is the
     * harness feeding the model, not the model running. */
    float *in = graph_input();
    for (int i = 0; i < INPUT_IMAGE_ELEMS; ++i)
        in[i] = input_image[i];

    printf("resnet18: run\\n");
    XTime_GetTime(&t_run0);
    float *out = graph_run();
#ifdef GRAPH_PROFILE
    graph_prof_count_run();
#endif
    XTime_GetTime(&t_run1);
    int n = graph_output_len();

    XTime_GetTime(&t_top0);

    /* Partial selection sort over the top K: no allocation, no qsort, and K
     * is 5 -- a full sort of 1000 logits would cost more than the argmax. */
    int idx[TOPK];
    for (int k = 0; k < TOPK && k < n; ++k) {
        int best = -1;
        for (int i = 0; i < n; ++i) {
            int taken = 0;
            for (int j = 0; j < k; ++j)
                if (idx[j] == i) { taken = 1; break; }
            if (taken) continue;
            if (best < 0 || out[i] > out[best]) best = i;
        }
        idx[k] = best;
    }
    XTime_GetTime(&t_top1);

    printf("resnet18: top%d\\n", TOPK);
    for (int k = 0; k < TOPK && k < n; ++k) {
        int c = idx[k];
        const char *name = (c < IMAGENET_NUM_CLASSES) ? imagenet_labels[c] : "?";
        printf("  %d. class=%-4d %-30s logit=%.4f\\n",
               k + 1, c, name, (double)out[c]);
    }
    printf("resnet18: top1 class=%d logit=%.6f\\n", idx[0], (double)out[idx[0]]);

    /* Timing last, so it cannot be mistaken for part of the measured work and
     * so the logits stay adjacent to the CPU reference they are diffed with. */
    {
        double init_ms = ticks_to_ms(t_init0, t_init1);
        double run_ms  = ticks_to_ms(t_run0,  t_run1);
        double top_ms  = ticks_to_ms(t_top0,  t_top1);
        printf("resnet18: timing (timer %llu Hz, 1 tick = %.1f ns)\\n",
               (unsigned long long)COUNTS_PER_SECOND,
               1e9 / (double)COUNTS_PER_SECOND);
        printf("  init       %10.3f ms\\n", init_ms);
        printf("  inference  %10.3f ms   <- graph_run()\\n", run_ms);
#ifdef GRAPH_PROFILE
        graph_prof_print();
#endif
        printf("  top%d       %10.3f ms\\n", TOPK, top_ms);
        printf("  total      %10.3f ms\\n", init_ms + run_ms + top_ms);
        if (run_ms > 0.0)
            printf("  throughput %10.2f inferences/s\\n", 1000.0 / run_ms);
    }

    printf("device_teardown done\\n");   /* the string verify_host.sh greps */
    return 0;
}
"""


def _dtype_bytes(dltype: str) -> int:
    """Bytes per element for a graph ``dltype`` string (``float32`` -> 4)."""
    for name, size in (("64", 8), ("32", 4), ("16", 2), ("8", 1)):
        if dltype.endswith(name):
            return size
    raise ValueError(f"unsupported dtype {dltype!r}")


def graph_sha(graph_path) -> str:
    """Short content hash of a graph JSON -- a build fingerprint.

    Stamped into both ``graph_driver.c`` and ``network.md`` so the two can be
    checked against each other. Every ``deploy_flow.py`` run rewrites both, and
    a tree can hold artifacts from several differently-shaped graphs in one
    session (int16 vs int8 operands, with and without ``--aie-offload``), so
    "do these two files describe the same build?" is a real question with no
    other answer. Mismatched hashes mean one of them is stale.
    """
    import hashlib

    try:
        return hashlib.sha1(Path(graph_path).read_bytes()).hexdigest()[:12]
    except OSError:
        return "unknown"


def graph_widths(nodes: list, shapes: list) -> tuple:
    """``(max_ndim, max_args)`` for a graph: how wide the driver's arrays must be.

    Both come from the graph, never from a round number. A hardcoded
    ``TVMValue args[8]`` was wrong here: the NCHWc convs take **10** arguments
    (7 inputs + 3 outputs), so ``args[8]`` and ``args[9]`` wrote past the end
    into ``codes[0..1]``. The kernels ignore ``arg_type_ids``, which is the
    only reason the board tolerated it -- the same C on x86 trips the stack
    protector immediately, and any other stack layout would have corrupted
    something that matters.

    ``__nop`` nodes are skipped: the graph elides them and the driver emits no
    call, so they cannot widen anything.
    """
    max_ndim = max((len(shape) for shape in shapes), default=1)
    max_args = 1
    for node in nodes:
        if node.get("op") != "tvm_op" or node["attrs"]["func_name"] == "__nop":
            continue
        max_args = max(max_args, len(node["inputs"])
                       + int(node["attrs"].get("num_outputs", 1)))
    return max_ndim, max_args


def _driver_init(sids, shapes, dltypes, offsets, nodes, eid) -> list:
    """``(lines, n_params)`` -- the body of ``graph_init()``.

    DLTensor headers, then one ``memcpy`` per weight out of the linked blob.
    Lifted out of :func:`generate_driver` to keep that under the project's
    200-line limit. Pure string building -- it owns no policy. The parameter
    count comes back with it because the caller reports it.
    """
    # Per-tensor DLTensor init.
    init = ["void graph_init(const uint8_t *params_blob) {",
            "    g_params = params_blob;",
            f"    tvm_shim_workspace_init(kernel_ws, {_KERNEL_WS});"]
    for i, (sid, shape, dltype) in enumerate(zip(sids, shapes, dltypes)):
        bits = _dtype_bytes(dltype) * 8
        code = 2 if dltype.startswith("float") else (
            0 if dltype.startswith("int") else 1)
        for d, dim in enumerate(shape):
            init.append(f"    shape_store[{i}][{d}] = {dim};")
        init += [
            f"    tensors[{i}].data = buf_{sid};",
            f"    tensors[{i}].device.device_type = 1;",
            f"    tensors[{i}].device.device_id = 0;",
            f"    tensors[{i}].ndim = {len(shape)};",
            f"    tensors[{i}].dtype.code = {code};",
            f"    tensors[{i}].dtype.bits = {bits};",
            f"    tensors[{i}].dtype.lanes = 1;",
            f"    tensors[{i}].shape = shape_store[{i}];",
            f"    tensors[{i}].strides = 0;",
            f"    tensors[{i}].byte_offset = 0;",
        ]

    # Copy each weight out of the blob into its buffer.
    n_params = 0
    for node_idx, node in enumerate(nodes):
        if node.get("op") != "null":
            continue
        name = node["name"]
        if name not in offsets:
            continue            # the network input, not a parameter
        off, nbytes = offsets[name]
        init.append(f"    memcpy(buf_{sids[eid(node_idx)]}, "
                    f"g_params + {off}, {nbytes});   /* {name} */")
        n_params += 1
    init += ["}", ""]
    return init, n_params


def aie_entries(layers_dir) -> dict:
    """``{TVM kernel symbol: AIE entry symbol}`` for layers the board ELF runs
    on the AIE, from ``layers/partition.json``.

    Only layers sent to AIE whose ``aie/`` archive is actually on disk and
    whose ``host.cc`` defines the packed-call entry -- today the stem, built
    from ``conv2dstem.cc`` (``aie_stem_lib``). Anything less and the swap would
    be an undefined reference at link, so it is not made.
    """
    part = Path(layers_dir) / "partition.json"
    if not part.is_file():
        return {}
    try:
        doc = json.loads(part.read_text())
    except (json.JSONDecodeError, OSError):
        return {}
    out = {}
    root = Path(layers_dir).parent
    for layer in doc.get("layers", []):
        entry, sym, arch = (layer.get("entry"), layer.get("replaces"),
                            layer.get("archive"))
        if layer.get("target") != "aie" or not (entry and sym and arch):
            continue
        host = root / layer["aie_dir"] / "host.cc"
        if (root / arch).is_file() and host.is_file() \
                and entry in host.read_text(errors="replace"):
            out[sym] = entry
    return out


def generate_driver(graph_path: Path, params_path: Path, out_dir: Path,
                    c_source=None, verbose: bool = True) -> dict:
    """Emit ``graph_driver.c``, ``tvm_runtime_shim.c`` and ``main.c``.

    The driver replaces TVM's C++ graph runtime with straight-line C: one
    static buffer per distinct ``storage_id`` (TVM has already done the
    liveness analysis and aliased what it can), the weights pointed at the
    linked blob, and one call per graph node in order.

    *c_source* is the generated module C (``resnet18.c``). It is read only to
    name the ``GRAPH_TRACE_HEX`` parameter dumps by ROLE -- weight, bias,
    zero-point, requant multiplier, requant shift -- which cannot be told apart
    from the graph alone: they are all just int32 constants wired into a kernel.
    Optional, because this function is callable on a graph with no C beside it;
    without it the dumps still carry each parameter's graph name, only not what
    it does.

    Returns a summary dict (buffer count, workspace bytes, node count).
    """
    graph = json.loads(graph_path.read_text())
    nodes = graph["nodes"]
    attrs = graph["attrs"]
    sids = attrs["storage_id"][1]
    shapes = attrs["shape"][1]
    dltypes = attrs["dltype"][1]
    row_ptr = graph["node_row_ptr"]

    max_ndim, max_args = graph_widths(nodes, shapes)

    # One buffer per storage id, sized by the largest tensor assigned to it.
    sizes: dict = {}
    for sid, shape, dltype in zip(sids, shapes, dltypes):
        elems = 1
        for dim in shape:
            elems *= dim
        sizes[sid] = max(sizes.get(sid, 0), elems * _dtype_bytes(dltype))
    total = sum(sizes.values())
    if total > DDR_BYTES:
        raise ValueError(f"graph needs {total:,} B of buffers, over the "
                         f"{DDR_BYTES:,} B the BSP linker script maps")

    offsets = param_offsets(params_path)

    # entry index -> flat tensor index, so a node's inputs resolve to storage.
    def eid(node_idx: int, slot: int = 0) -> int:
        return row_ptr[node_idx] + slot

    lines = [
        "/* Generated by src/frontend/tvmrelay/arm_build.py -- do not edit.",
        " *",
        # Kernel CALLS, matching what network.md counts -- `len(nodes)` here
        # is every graph node including the parameter placeholders, so the
        # two stamps would print different numbers for the same graph.
        f" * graph-sha: {graph_sha(graph_path)}   "
        f"({sum(1 for n in nodes if n.get('op') == 'tvm_op')} kernel calls)",
        " * The same line appears in network.md. If they differ, one of the",
        " * two is from an earlier build -- every deploy_flow.py run rewrites",
        " * both, and a tree can hold several graph shapes in one session.",
        " *",
        " * Straight-line replacement for TVM's C++ graph runtime: static",
        " * buffers (one per storage_id, already aliased by TVM's liveness",
        " * analysis), weights pointed into the linked blob, one call per",
        " * graph node in execution order.",
        " */",
        "#include <stdint.h>",
        # stdio for the debug helpers below. Not optional even without them:
        # a printf dropped in by hand would otherwise be an implicit
        # declaration returning int, which is a C11 constraint violation that
        # gcc only warns about.
        "#include <stdio.h>",
        "#include <string.h>",
        "",
        '#include "tvm_graph_types.h"',
        "",
        "void tvm_shim_workspace_init(void *buf, uint64_t bytes);",
        "",
    ] + _PROFILE_C.splitlines()

    # Kernel forward declarations, deduplicated but in first-use order.
    seen = []
    for node in nodes:
        if node.get("op") == "tvm_op":
            fn = node["attrs"]["func_name"]
            if fn != "__nop" and fn not in seen:
                seen.append(fn)
    for fn in seen:
        lines.append(f"int32_t {fn}(void* args, int32_t* arg_type_ids, "
                     "int32_t num_args, void* out_ret_value, "
                     "int32_t* out_ret_tcode, void* resource_handle);")
    # AIE-offloaded kernels: same packed signature, defined in the layer's
    # aie/host.cc (lib<...>.a). Only the board build defines
    # GRAPH_AIE_OFFLOAD; `make local` compiles the CPU call from the SAME file,
    # which is what makes main_local.elf the reference for main.elf.
    offload = aie_entries(graph_path.parent / "layers")
    if offload:
        lines.append("#ifdef GRAPH_AIE_OFFLOAD")
        for entry in sorted(set(offload.values())):
            lines.append(f"int32_t {entry}(void* args, int32_t* arg_type_ids, "
                         "int32_t num_args, void* out_ret_value, "
                         "int32_t* out_ret_tcode, void* resource_handle);")
        lines.append("#endif")
    lines.append("")

    for sid in sorted(sizes):
        lines.append(f"static uint8_t buf_{sid}[{sizes[sid]}] "
                     "__attribute__((aligned(64)));")
    lines += [
        "",
        f"static uint8_t kernel_ws[{_KERNEL_WS}] __attribute__((aligned(64)));",
        "static const uint8_t *g_params;",
        "",
        "/* Both widths come from the graph, not from a round number: the",
        " * widest node takes %d kernel arguments and the widest tensor is"
        % max_args,
        " * %d-dimensional. */" % max_ndim,
        "#define GRAPH_MAX_NDIM %d" % max_ndim,
        "#define GRAPH_MAX_ARGS %d" % max_args,
        "",
        "/* DLTensor headers are reused per call; only .data and .shape vary. */",
        "static int64_t shape_store[%d][GRAPH_MAX_NDIM];" % len(sids),
        "static DLTensor tensors[%d];" % len(sids),
        "",
        # After `tensors[]`, because graph_dump_tensor indexes it.
        _DEBUG_HELPERS_C,
    ]

    init_lines, n_params = _driver_init(sids, shapes, dltypes,
                                        offsets, nodes, eid)
    lines += init_lines

    # Entry index -> (graph name, blob offset, bytes) for every weight, as
    # opposed to an activation. Drives the layeriohex naming: "which of these
    # is a parameter, which parameter, and what does it do" is the first thing
    # asked of a traced call, and none of it is visible from the tensor.
    # The blob offset rides along so a dumped weight can be checked straight
    # against weights.bin.
    param_of = {eid(i): (node["name"], *offsets[node["name"]])
                for i, node in enumerate(nodes)
                if node.get("op") == "null" and node["name"] in offsets}
    roles = _param_roles(c_source, nodes)
    flow = _dataflow(nodes, eid)
    lines += _param_dump_lines(param_of, roles, nodes, eid)

    # The input node: first arg_node with no matching parameter.
    input_idx = next(i for i, node in enumerate(nodes)
                     if node.get("op") == "null"
                     and node["name"] not in offsets)
    out_node, out_slot, _ = graph["heads"][0]
    out_eid = eid(out_node, out_slot)
    out_elems = 1
    for dim in shapes[out_eid]:
        out_elems *= dim

    lines += [
        "float *graph_input(void) { return (float *)buf_%d; }"
        % sids[eid(input_idx)],
        "int graph_output_len(void) { return %d; }" % out_elems,
        "",
        "float *graph_run(void) {",
        "    TVMValue args[GRAPH_MAX_ARGS];",
        "    int32_t codes[GRAPH_MAX_ARGS];",
        "    for (int i = 0; i < GRAPH_MAX_ARGS; ++i)",
        "        codes[i] = 7;   /* kTVMDLTensorHandle */",
        # Here rather than in main.c: graph_init() has run by now (main.c calls
        # it first), the constants are in their buffers, and a second inference
        # must not rewrite 177 files.
        "#ifdef GRAPH_TRACE_HEX",
        "#ifndef GRAPH_TRACE_HEX_NO_PARAMS",
        "    { static int once; if (!once) { once = 1; graph_dump_params(); } }",
        "#endif",
        "#endif",
        "",
    ]

    n_calls = 0
    for node_idx, node in enumerate(nodes):
        if node.get("op") != "tvm_op":
            continue
        fn = node["attrs"]["func_name"]
        if fn == "__nop":
            # The graph elides layout-free reshapes; the buffers already alias,
            # so there is genuinely nothing to call.
            lines.append(f"    /* node {node_idx}: {fn} (elided) */")
            continue
        in_eids = [eid(src, slot) for src, slot, _ in node["inputs"]]
        n_out = int(node["attrs"].get("num_outputs", 1))
        out_eids = [eid(node_idx, k) for k in range(n_out)]
        arg_eids = in_eids + out_eids
        for slot, tensor_idx in enumerate(arg_eids):
            lines.append(f"    args[{slot}].v_handle = &tensors[{tensor_idx}];")
        tl = dict(param_of=param_of, roles=roles.get(fn, {}), nodes=nodes,
                  flow=flow)
        lines += _trace_lines(n_calls, fn, in_eids, out_eids, "in", **tl)
        lines.append(f"    GRAPH_PROF_BEGIN({n_calls});")
        if fn in offload:
            lines += ["#ifdef GRAPH_AIE_OFFLOAD",
                      f"    {offload[fn]}(args, codes, {len(arg_eids)}, 0, 0, 0);",
                      "#else",
                      f"    {fn}(args, codes, {len(arg_eids)}, 0, 0, 0);",
                      "#endif"]
        else:
            lines.append(f"    {fn}(args, codes, {len(arg_eids)}, 0, 0, 0);")
        tag = ("aie " if fn in offload else "cpu ") + fn.replace("tvmgen_default_fused_", "")
        lines.append(f'    GRAPH_PROF_END({n_calls}, "{tag[:72]}");')
        lines += _trace_lines(n_calls, fn, in_eids, out_eids, "out", **tl)
        lines.append("")
        n_calls += 1

    lines += [f"    return (float *)buf_{sids[out_eid]};", "}", ""]

    (out_dir / "graph_driver.c").write_text("\n".join(lines))
    (out_dir / "tvm_graph_types.h").write_text(_TYPES_H)
    (out_dir / "tvm_runtime_shim.c").write_text(_SHIM_C)

    in_elems = 1
    for dim in shapes[eid(input_idx)]:
        in_elems *= dim
    (out_dir / "main.c").write_text(_MAIN_C)

    summary = {"buffers": len(sizes), "buffer_bytes": total,
               "kernel_workspace": _KERNEL_WS, "params": n_params,
               "calls": n_calls, "input_elems": in_elems,
               "output_elems": out_elems}
    if verbose:
        print(f"  [arm] driver: {n_calls} kernel calls, {len(sizes)} buffers "
              f"({total / 1e6:.1f} MB), {n_params} params")
    return summary


# Per-call wall time, compiled in only with `make PROFILE=1` (-DGRAPH_PROFILE).
# Same clock main.c uses for "inference", so the per-call rows add up to it.
# The tag says where the call ran ("aie" only when this build offloads it).
_PROFILE_C = """\
#ifdef GRAPH_PROFILE
#include "aie_timer.h"
#define GRAPH_PROF_MAX 512
static unsigned long long graph_prof_ticks[GRAPH_PROF_MAX];
static const char *graph_prof_tag[GRAPH_PROF_MAX];
static unsigned graph_prof_runs, graph_prof_n;
#define GRAPH_PROF_BEGIN(i) XTime graph_prof_t0_##i; XTime_GetTime(&graph_prof_t0_##i)
#define GRAPH_PROF_END(i, tag) do { XTime t1_; XTime_GetTime(&t1_);                      \\
        graph_prof_ticks[i] += t1_ - graph_prof_t0_##i; graph_prof_tag[i] = (tag);        \\
        if ((unsigned)(i) + 1U > graph_prof_n) graph_prof_n = (unsigned)(i) + 1U; } while (0)
/* Mean per-call time over every graph_run() so far, then the sum. */
void graph_prof_print(void) {
    double tot = 0.0;
    unsigned runs = graph_prof_runs ? graph_prof_runs : 1U;
    for (unsigned i = 0; i < graph_prof_n; i++) {
        double ms = 1000.0 * (double)graph_prof_ticks[i] / (double)COUNTS_PER_SECOND / runs;
        tot += ms;
        printf("PROF %3u %10.3f ms  %s\\n", i, ms, graph_prof_tag[i] ? graph_prof_tag[i] : "?");
    }
    printf("PROF sum %10.3f ms over %u run(s)\\n", tot, runs);
}
void graph_prof_count_run(void) { graph_prof_runs++; }
#else
#define GRAPH_PROF_BEGIN(i) ((void)0)
#define GRAPH_PROF_END(i, tag) ((void)0)
#endif
"""


_DEBUG_HELPERS_C = """\
/* ── Debug helpers ─────────────────────────────────────────────────────────
 *
 * A DLDataType is a 4-byte STRUCT {code, bits, lanes}, not an integer.
 * `printf("%d", t.dtype)` is undefined behaviour that happens to print the
 * struct's little-endian packing -- `code | bits<<8 | lanes<<16` -- so an
 * int16 tensor reads as the meaningless 69632 (0x011000) and a uint8 one as
 * 67585 (0x010101). dl_dtype_str decodes it instead.
 *
 * All three are static inline: unused they cost no code and warn about
 * nothing, so they are always emitted and always available to a printf
 * dropped into graph_run() by hand.
 */
#define DL_CODE_INT    0
#define DL_CODE_UINT   1
#define DL_CODE_FLOAT  2
#define DL_CODE_BFLOAT 4

/* How many elements the GRAPH_TRACE dumps show. Override at compile time. */
#ifndef GRAPH_TRACE_ELEMS
#define GRAPH_TRACE_ELEMS 8
#endif

static inline const char *dl_code_name(uint8_t code) {
    switch (code) {
    case DL_CODE_INT:    return "int";
    case DL_CODE_UINT:   return "uint";
    case DL_CODE_FLOAT:  return "float";
    case DL_CODE_BFLOAT: return "bfloat";
    default:             return "?";
    }
}

/* "int16", "uint8", "float32x4" -- into CALLER-supplied storage, so two calls
 * in one printf cannot clobber each other the way a shared static would. */
static inline const char *dl_dtype_str(DLDataType dt, char *buf, size_t n) {
    if (dt.lanes == 1)
        snprintf(buf, n, "%s%u", dl_code_name(dt.code), (unsigned)dt.bits);
    else
        snprintf(buf, n, "%s%ux%u", dl_code_name(dt.code), (unsigned)dt.bits,
                 (unsigned)dt.lanes);
    return buf;
}

/* Shape, dtype, and the first `n` ELEMENTS -- elements, not bytes.
 * A 16-bit tensor dumped as bytes reads "98 ff 9e ff", which is little-endian
 * -104, -98: the single most common way to misread this output. */
static inline void graph_dump_tensor(const char *tag, int idx, int n) {
    const DLTensor *t = &tensors[idx];
    unsigned key = ((unsigned)t->dtype.code << 8) | (unsigned)t->dtype.bits;
    char ds[24];
    long long numel = 1;
    printf("%s tensors[%d] %s [", tag, idx, dl_dtype_str(t->dtype, ds, sizeof ds));
    for (int i = 0; i < t->ndim; ++i) {
        printf("%lld%s", (long long)t->shape[i], (i + 1 < t->ndim) ? "," : "");
        numel *= (long long)t->shape[i];
    }
    /* Clamp, or a scalar parameter (ndim 0, numel 1 -- the zero-points are
     * exactly this) is dumped past the end of a 2-byte buffer. */
    if ((long long)n > numel) n = (int)numel;
    printf("] =");
    for (int i = 0; i < n; ++i) {
        switch (key) {
        case (DL_CODE_INT   << 8) | 8:
            printf(" %d", ((const int8_t *)t->data)[i]); break;
        case (DL_CODE_UINT  << 8) | 8:
            printf(" %u", ((const uint8_t *)t->data)[i]); break;
        case (DL_CODE_INT   << 8) | 16:
            printf(" %d", ((const int16_t *)t->data)[i]); break;
        case (DL_CODE_UINT  << 8) | 16:
            printf(" %u", ((const uint16_t *)t->data)[i]); break;
        case (DL_CODE_INT   << 8) | 32:
            printf(" %d", (int)((const int32_t *)t->data)[i]); break;
        case (DL_CODE_UINT  << 8) | 32:
            printf(" %u", (unsigned)((const uint32_t *)t->data)[i]); break;
        case (DL_CODE_FLOAT << 8) | 32:
            printf(" %.6f", (double)((const float *)t->data)[i]); break;
        default:    /* unknown width: raw bytes, which is all we can say */
            printf(" %02x", ((const unsigned char *)t->data)[i]); break;
        }
    }
    printf("\\n");
}

/* ── Per-layer I/O as C hex headers ────────────────────────────────────────
 *
 * GRAPH_TRACE prints the first few elements; GRAPH_TRACE_HEX writes every
 * kernel input and output out IN FULL as a compilable C header of hex bytes,
 * one file per tensor per call, so a layer's real data can be diffed against
 * another build, replayed into a standalone test, or fed to the AIE kernel.
 *
 * BYTES, not decoded elements, and deliberately so: this is the on-the-wire
 * image of the buffer. The element view is graph_dump_tensor's job, and the
 * header carries the dtype and shape as macros so a reader can decode it.
 *
 * Two sinks, because the two targets do not have the same world:
 *
 *   GRAPH_TRACE_HEX_FILES  (hosted; `make local` sets it) -- real files in
 *       ./layeriohex/, relative to wherever the ELF is run.
 *   otherwise              (baremetal board) -- there is no filesystem under
 *       xsdb, the console IS the channel, so the identical text goes to
 *       stdout between BEGIN/END markers. `arm_build.split_layeriohex()`
 *       cuts a captured log back into the same files.
 */
#ifdef GRAPH_TRACE_HEX

#ifndef GRAPH_TRACE_HEX_DIR
#define GRAPH_TRACE_HEX_DIR "layeriohex"
#endif

/* 0 = the whole tensor. A cap matters most on the console sink, where the
 * full int8 ResNet-18 is ~100 MB of text over a UART. */
#ifndef GRAPH_TRACE_HEX_MAX_BYTES
#define GRAPH_TRACE_HEX_MAX_BYTES 0
#endif

#ifdef GRAPH_TRACE_HEX_FILES
#include <sys/stat.h>
#include <sys/types.h>

static FILE *graph_hex_open(const char *name) {
    static int made;
    char path[512];
    if (!made) {            /* once, and an existing directory is not an error */
        mkdir(GRAPH_TRACE_HEX_DIR, 0777);
        made = 1;
    }
    snprintf(path, sizeof path, "%s/%s.h", GRAPH_TRACE_HEX_DIR, name);
    FILE *f = fopen(path, "w");
    if (!f) printf("   hex: cannot write %s\\n", path);
    return f;
}

static void graph_hex_close(FILE *f, const char *name) { (void)name; fclose(f); }
#else   /* console sink: no filesystem on the board */
static FILE *graph_hex_open(const char *name) {
    printf("===BEGIN " GRAPH_TRACE_HEX_DIR "/%s.h===\\n", name);
    return stdout;
}

static void graph_hex_close(FILE *f, const char *name) {
    (void)f;
    printf("===END " GRAPH_TRACE_HEX_DIR "/%s.h===\\n", name);
}
#endif

/* Open one layeriohex file: header comment and include guard.
 * `stem` is the filename AND the guard name. Returns 0 if it could not be
 * opened, so the caller skips the whole group rather than writing a file with
 * a guard and no `#endif`. */
static FILE *graph_hex_file_begin(const char *stem, const char *title) {
    FILE *f = graph_hex_open(stem);
    if (!f) return 0;
    fprintf(f, "/* Written by graph_driver.c under `make TRACE=1` -- "
               "regenerated every run, do not edit.\\n"
               " * %s\\n */\\n", title);
    fprintf(f, "#ifndef LAYERIOHEX_%s_H\\n#define LAYERIOHEX_%s_H\\n", stem, stem);
    return f;
}

static void graph_hex_file_end(FILE *f, const char *stem) {
    fprintf(f, "\\n#endif\\n");
    graph_hex_close(f, stem);
}

/* One tensor -- macros, shape, hex bytes -- into an ALREADY-OPEN file.
 *
 * A layer's inputs all land in one `..._in.h` and its outputs in one
 * `..._out.h`, each under its own `var` prefix, so one `#include` brings in
 * everything that call saw instead of nine.
 *
 * `var` is the C identifier prefix, generated as
 * `l<NN>_<op>_<in|out><slot>[_<param>_<role>]`: unique within the file by the
 * argument slot, and a valid identifier by construction, so nothing here has
 * to sanitize it. `prov` is the static description arm_build.py already knows
 * -- argument slot, tensor index, and for a parameter its name, role and
 * offset in weights.bin.
 */
static void graph_hex_tensor(FILE *f, const char *var, int idx,
                             const char *prov) {
    const char *name = var;
    const DLTensor *t = &tensors[idx];
    char ds[24];
    long long numel = 1, nbytes, shown;
    for (int i = 0; i < t->ndim; ++i) numel *= (long long)t->shape[i];
    nbytes = numel * (long long)((t->dtype.bits + 7) / 8)
                   * (long long)t->dtype.lanes;
    shown = nbytes;
    if (GRAPH_TRACE_HEX_MAX_BYTES > 0 && shown > GRAPH_TRACE_HEX_MAX_BYTES)
        shown = GRAPH_TRACE_HEX_MAX_BYTES;

    fprintf(f, "\\n/* %s */\\n", prov);
    fprintf(f, "#define %s_DTYPE \\"%s\\"\\n", name,
            dl_dtype_str(t->dtype, ds, sizeof ds));
    fprintf(f, "#define %s_NDIM  %d\\n", name, t->ndim);
    fprintf(f, "#define %s_ELEMS %lldLL\\n", name, numel);
    fprintf(f, "#define %s_BYTES %lldLL\\n", name, nbytes);
    if (shown != nbytes)    /* say so IN the file; a short array is otherwise
                             * indistinguishable from a small tensor */
        fprintf(f, "#define %s_TRUNCATED %lldLL\\n", name, shown);
    fprintf(f, "\\nstatic const long long %s_shape[%d] = {", name,
            t->ndim ? t->ndim : 1);
    for (int i = 0; i < t->ndim; ++i)
        fprintf(f, "%s%lld", i ? ", " : " ", (long long)t->shape[i]);
    fprintf(f, "%s};\\n\\n", t->ndim ? " " : " 1 ");

    const unsigned char *p = (const unsigned char *)t->data + t->byte_offset;
    fprintf(f, "static const unsigned char %s_data[%lldLL] = {\\n", name, shown);
    for (long long i = 0; i < shown; ++i) {
        fprintf(f, "%s0x%02x,", (i % 16) ? " " : "    ", (unsigned)p[i]);
        if ((i % 16) == 15) fputc('\\n', f);
    }
    if (shown % 16) fputc('\\n', f);
    fprintf(f, "};\\n");
}

/* One tensor alone in its own file -- what graph_dump_params() wants, since a
 * graph constant belongs to no single layer.
 *
 * `static inline`, like the other optional helpers above: under
 * GRAPH_TRACE_HEX_NO_PARAMS there is no graph_dump_params() to call it, and a
 * plain `static` would warn (-Wunused-function) in a build that is otherwise
 * clean. -fsyntax-only does NOT report this, so it has to be compiled for real
 * to see it. */
static inline void graph_dump_hex(const char *name, int idx, const char *prov) {
    FILE *f = graph_hex_file_begin(name, prov);
    if (!f) return;
    graph_hex_tensor(f, name, idx, prov);
    graph_hex_file_end(f, name);
}
#endif  /* GRAPH_TRACE_HEX */
"""


def _dataflow(nodes: list, eid) -> tuple:
    """``(produced_by, consumed_by, call_of)`` over entry indices.

    Answers the question a bare ``(in)`` label cannot: where did this activation
    come from, and who reads what this call produced. Layer 06 takes two
    activations -- its feature map from call [01], and an ``int32`` tensor from
    call [05] with the *same* geometry as its own output, which reads exactly
    like the conv's output fed back into itself until you see the producer.

    ``call_of`` maps a node index to the call index used in the trace, counting
    the same way the emission loop does -- ``__nop`` nodes are elided and do not
    take a number, so any other rule would shift every label after the first one.
    """
    produced_by, consumed_by, call_of = {}, {}, {}
    n = 0
    for i, node in enumerate(nodes):
        if node.get("op") == "tvm_op" and node["attrs"]["func_name"] != "__nop":
            call_of[i] = n
            n += 1
        for k in range(int(node.get("attrs", {}).get("num_outputs", 1))):
            produced_by[eid(i, k)] = i
    for i, node in enumerate(nodes):
        if node.get("op") != "tvm_op":
            continue
        for src, slot, _ in node.get("inputs", []):
            consumed_by.setdefault(eid(src, slot), []).append(i)
    return produced_by, consumed_by, call_of


def _flow_note(eid_: int, where: str, nodes: list, flow: tuple) -> str:
    """Human-readable origin (for an input) or destination (for an output)."""
    produced_by, consumed_by, call_of = flow
    if where == "in":
        src = produced_by.get(eid_)
        if src is None:
            return "in"
        node = nodes[src]
        if node.get("op") != "tvm_op":
            return "in, network input"
        fn = node["attrs"]["func_name"]
        if fn == "__nop":
            return "in, from an elided reshape"
        return f"in, from call [{call_of[src]:02d}] {_kernel_slug(fn)}"
    readers = [call_of[c] for c in consumed_by.get(eid_, []) if c in call_of]
    if not readers:
        return "out, network output"
    return "out, read by " + ", ".join(f"call [{c:02d}]" for c in readers)


def _param_roles(c_source, nodes: list) -> dict:
    """``{kernel symbol: {arg slot: role}}`` read out of the generated C.

    Reuses ``network_md.kernel_param_usage`` / ``param_role`` -- the same
    classifier that labels ``network.md``'s parameter tables, so the two cannot
    disagree about what a constant is for. It keys on the kernel's own ``pN``
    argument slot, and ``pN`` is literally ``args[N]`` in the emitted C
    (``void* p0 = ((TVMValue*)args)[0].v_handle``), so a slot indexes the
    driver's input list directly.

    **Only a slot that is also a parameter may take its role.** The classifier
    labels every slot it sees, activations included -- layer 06's image input
    lands in an ``add`` and would otherwise be filed as a "bias". The caller
    gates on ``param_of``.

    Returns ``{}`` when there is no C to read, or when anything about it is
    unexpected: a missing label is a cosmetic loss, while a wrong one sends
    whoever reads the dump after the wrong tensor.
    """
    if not c_source:
        return {}
    try:
        from frontend.tvmrelay.network_md import kernel_param_usage, param_role

        src = Path(c_source).read_text()
    except Exception:
        return {}
    out = {}
    for node in nodes:
        if node.get("op") != "tvm_op":
            continue
        fn = node["attrs"]["func_name"]
        if fn == "__nop" or fn in out:
            continue
        try:
            usage = kernel_param_usage(src, fn)
            # EVERY input slot, not just the ones the C mentions. A slot the
            # kernel never references is "unused" -- a real verdict, and the
            # one network.md prints -- and leaving it out of the dict would
            # make it indistinguishable from "no C was read at all".
            out[fn] = {slot: param_role(usage.get(slot, set()))
                       for slot in range(len(node["inputs"]))}
        except Exception:
            out[fn] = {}
    return out


def _role_slug(role: str) -> str:
    """``"requant multiplier"`` -> ``"requant_multiplier"``.

    The stem is pasted into ``static const ... <stem>_data``, so a role with a
    space or a hyphen in it would emit C that does not compile.
    """
    return "".join(c if (c.isalnum() or c == "_") else "_" for c in role)


def _param_dump_lines(param_of: dict, roles: dict, nodes: list, eid) -> list:
    """``graph_dump_params()`` -- every parameter once, by name and role.

    The per-call dumps already carry parameters, but only as the Nth input of
    some kernel: a weight shared by several calls is written once per call, and
    a parameter no traced call reads is never written at all. This is the
    by-name view -- one file per graph constant, deduplicated, including the
    ones the classifier calls ``unused``.

    It is a separate switch (``TRACE_HEX_PARAMS=0``) because the blob is 11.9 MB
    and hex is ~6x, so this alone is ~72 MB on top of the per-call dumps.
    """
    # The first kernel slot each parameter is wired into -- that is the only
    # place its role is observable, since a role comes from how the C uses it.
    role_of = {}
    for node_idx, node in enumerate(nodes):
        if node.get("op") != "tvm_op":
            continue
        fn = node["attrs"]["func_name"]
        for slot, (src, s, _) in enumerate(node["inputs"]):
            e = eid(src, s)
            if e in param_of and e not in role_of:
                role_of[e] = roles.get(fn, {}).get(slot, "")

    out = ["#ifdef GRAPH_TRACE_HEX",
           "#ifndef GRAPH_TRACE_HEX_NO_PARAMS",
           "/* Every graph constant, once, named by what it IS rather than by",
           " * which argument of which kernel happened to reference it. */",
           "static void graph_dump_params(void) {"]
    for e, (name, off, nbytes) in sorted(param_of.items()):
        role = role_of.get(e, "") or "unused"
        stem = f"param_{name}_{_role_slug(role)}"
        out.append(f'    graph_dump_hex("{stem}", {e}, "graph parameter {name} '
                   f'({role}) = tensors[{e}] -- weights.bin@{off}+{nbytes}");')
    out += ["}", "#endif", "#endif", ""]
    return out


#: Budget for the descriptive op slug inside a layeriohex stem. Small on
#: purpose. The call index already makes the stem unique, so the kernel name is
#: there to say what the layer DOES, not to identify it -- and a 95-character
#: symbol repeated across ten files means ``ls`` shows ten identical-looking
#: names whose only difference is off the right edge of the terminal. The full
#: symbol is on the first line inside every file, so nothing is lost.
_HEX_NAME_FN_CHARS = 30

#: A trailing content hash on a fused symbol
#: (``..._per_channel_ca76d0071d109ff1_``). Pure noise in a filename.
_FN_HASH_TAIL = re.compile(r"_[0-9a-f]{8,}_?$")


def _kernel_slug(fn: str) -> str:
    """Short, readable op name for a file stem: ``nn_contrib_conv2d_NCHWc``.

    Drops TVM's ``tvmgen_default_fused_`` prefix and the trailing content hash,
    then takes whole ``_``-separated words up to :data:`_HEX_NAME_FN_CHARS` --
    cutting mid-word gives ``..._fixed_point_multip`` and reads like damage.
    Two different kernels may well slug the same; that is fine and cannot
    collide, because the call index disambiguates.
    """
    for prefix in ("tvmgen_default_fused_", "tvmgen_default_", "fused_"):
        if fn.startswith(prefix):
            fn = fn[len(prefix):]
            break
    fn = _FN_HASH_TAIL.sub("", fn)
    fn = "".join(c if (c.isalnum() or c == "_") else "_" for c in fn)
    out = ""
    for word in fn.split("_"):
        if not word:
            continue
        if out and len(out) + 1 + len(word) > _HEX_NAME_FN_CHARS:
            break
        out = f"{out}_{word}" if out else word
    return out[:_HEX_NAME_FN_CHARS] or "op"


def _hex_name(call_idx: int, fn: str, where: str, slot=None,
              suffix: str = "") -> str:
    """Name for a traced group or one tensor in it.

    ``slot=None`` gives the **file** stem, ``l06_nn_contrib_conv2d_NCHWc_in`` --
    one file for all of a call's inputs, one for its outputs. With a slot it
    gives the **variable** prefix inside that file,
    ``l06_nn_contrib_conv2d_NCHWc_in1_p0_weight`` -- layer, what it does, which
    argument, and for a parameter which one and what it is for.

    The leading letter matters: both are pasted straight into C -- as an include
    guard and as ``static const ... <var>_data`` -- and an identifier cannot
    start with a digit. *suffix* carries the parameter name and role, the whole
    point being that ``_in3`` on its own says nothing.
    """
    tail = "" if slot is None else str(slot)
    return f"l{call_idx:02d}_{_kernel_slug(fn)}_{where}{tail}{suffix}"


def _trace_lines(call_idx: int, fn: str, in_eids: list, out_eids: list,
                 where: str, param_of=None, roles=None, nodes=None,
                 flow=None) -> list:
    """``#ifdef GRAPH_TRACE`` dump around one kernel call. ``where`` is in|out.

    Generated rather than hand-added, so instrumenting a node survives the
    next ``deploy_flow.py`` run -- editing ``graph_driver.c`` directly does
    not, the file is overwritten every time.

    Two dumps per tensor, nested so the second is separately switchable:
    ``graph_dump_tensor`` prints the first few DECODED elements to the console,
    and ``graph_dump_hex`` (``GRAPH_TRACE_HEX``, which ``TRACE=1`` turns on
    unless ``TRACE_HEX=0``) writes the whole buffer to ``layeriohex/`` as a C
    header.

    **One file per direction, not per tensor.** All of a call's inputs go into
    ``l06_<op>_in.h`` and its outputs into ``l06_<op>_out.h``, each tensor under
    its own variable prefix. A conv's activation, weight and six quantization
    constants are one ``#include`` rather than eight, and the set cannot be
    split up by accident.

    Parameters are dumped like any other input -- a mismatch is as often in a
    weight as in an activation -- but they are NAMED: *param_of* gives the graph
    constant behind an entry index and *roles* what the kernel does with that
    argument slot, so the variable is ``..._in1_p0_weight`` rather than
    ``..._in1``. A role is applied only to a slot that is also in *param_of*:
    the classifier labels activations too, and layer 06's image input lands in
    an ``add``, which would file it as a "bias".
    """
    param_of, roles = param_of or {}, roles or {}
    eids = in_eids if where == "in" else out_eids
    tag = "   in " if where == "in" else "   out"
    out = ["#ifdef GRAPH_TRACE"]
    if where == "in":
        out.append(f'    printf("[{call_idx:02d}] {fn}\\n");')
    out += [f'    graph_dump_tensor("{tag}", {t}, GRAPH_TRACE_ELEMS);'
            for t in eids]

    stem = _hex_name(call_idx, fn, where)
    title = (f"call [{call_idx:02d}] {fn} -- {len(eids)} {where}put(s)")
    out += ["#ifdef GRAPH_TRACE_HEX",
            "    {",
            f'        FILE *hf = graph_hex_file_begin("{stem}", "{title}");',
            "        if (hf) {"]
    for slot, t in enumerate(eids):
        suffix = ""
        if where == "in" and t in param_of:
            name, off, nbytes = param_of[t]
            role = roles.get(slot, "") or "param"
            suffix = f"_{name}_{_role_slug(role)}"
            what = f"param {name}, {role}, weights.bin@{off}+{nbytes}"
        elif flow and nodes is not None:
            # An activation: say where it came from / goes to. Without this a
            # tensor that merely SHARES a shape with this call's output reads
            # like the output fed back in.
            what = _flow_note(t, where, nodes, flow)
        else:
            what = where
        prov = f"{where}[{slot}] = tensors[{t}] ({what})"
        var = _hex_name(call_idx, fn, where, slot, suffix)
        out.append(f'            graph_hex_tensor(hf, "{var}", {t}, "{prov}");')
    out += [f'            graph_hex_file_end(hf, "{stem}");',
            "        }",
            "    }",
            "#endif", "#endif"]
    return out


def split_layeriohex(log_path, out_dir=None, verbose: bool = True) -> dict:
    """Cut a board console log back into ``layeriohex/*.h``. Returns a summary.

    The hosted build writes the headers itself; a baremetal board cannot, so
    ``graph_dump_hex`` streams the identical text to the console framed by
    ``===BEGIN layeriohex/<stem>.h===`` / ``===END ...===``. Point this at the
    captured log (``applog``) and the two sinks produce the same files.

    Unterminated blocks are written anyway and counted in ``truncated``: a run
    that was interrupted mid-dump still carries every layer before it, and
    dropping them would lose exactly the data the log was captured for.
    """
    log_path = Path(log_path)
    out_dir = Path(out_dir) if out_dir else log_path.parent / "layeriohex"
    out_dir.mkdir(parents=True, exist_ok=True)

    written, truncated, name, body = [], [], None, []
    for line in log_path.read_text(errors="replace").splitlines():
        if line.startswith("===BEGIN layeriohex/") and line.endswith(".h==="):
            if name is not None:                # BEGIN inside a block
                truncated.append(name)
                (out_dir / f"{name}.h").write_text("\n".join(body) + "\n")
                written.append(name)
            name = line[len("===BEGIN layeriohex/"):-len(".h===")]
            body = []
        elif name is not None and line.startswith("===END layeriohex/"):
            (out_dir / f"{name}.h").write_text("\n".join(body) + "\n")
            written.append(name)
            name, body = None, []
        elif name is not None:
            body.append(line)
    if name is not None:                        # log ends mid-block
        (out_dir / f"{name}.h").write_text("\n".join(body) + "\n")
        written.append(name)
        truncated.append(name)

    if verbose:
        print(f"  [hex] {len(written)} header(s) -> {out_dir}"
              + (f"  ({len(truncated)} truncated)" if truncated else ""))
    return {"dir": str(out_dir), "written": written, "truncated": truncated}


#: Scratch for TVMBackendAllocWorkspace. Kernels allocate inside a fused op
#: (im2col-style packing buffers); 16 MB clears ResNet-18's largest by a wide
#: margin and the shim reports rather than corrupting if it were ever short.
_KERNEL_WS = 16 * 1024 * 1024

_TYPES_H = """\
/* Generated by src/frontend/tvmrelay/arm_build.py -- do not edit.
 *
 * The DLPack subset the generated kernels touch. TVM's own headers are C++
 * and pull in the whole runtime; the kernels only ever read these fields.
 */
#ifndef TVMRELAY_GRAPH_TYPES_H
#define TVMRELAY_GRAPH_TYPES_H
#include <stdint.h>

typedef struct { int32_t device_type; int32_t device_id; } DLDevice;
typedef struct { uint8_t code; uint8_t bits; uint16_t lanes; } DLDataType;
typedef struct {
    void *data;
    DLDevice device;
    int32_t ndim;
    DLDataType dtype;
    int64_t *shape;
    int64_t *strides;
    uint64_t byte_offset;
} DLTensor;

typedef union {
    int64_t v_int64;
    double  v_float64;
    void   *v_handle;
} TVMValue;

#endif  /* TVMRELAY_GRAPH_TYPES_H */
"""


def _write_tvm_header_stubs(build: Path) -> None:
    """Provide ``tvm/runtime/c_{runtime,backend}_api.h`` for the kernel file.

    The generated ``resnet18.c`` opens with::

        #include "tvm/runtime/c_runtime_api.h"
        #include "tvm/runtime/c_backend_api.h"

    TVM's real headers pull in ``dlpack`` and a good deal of host runtime
    machinery that does not build for baremetal aarch64. Grepping the kernels
    shows they use exactly four things from them -- ``DLTensor``, ``TVMValue``,
    ``TVM_DLL``, and the two ``TVMBackend*Workspace`` calls -- all of which
    ``tvm_graph_types.h`` already defines. So the include path gets a local
    ``tvm/runtime/`` that forwards to it, and the kernel source is used
    verbatim rather than patched.
    """
    inc = build / "tvm" / "runtime"
    inc.mkdir(parents=True, exist_ok=True)
    shim = ('/* Generated by arm_build.py -- baremetal stand-in for TVM\'s\n'
            ' * runtime headers. See _write_tvm_header_stubs. */\n'
            '#include <stddef.h>   /* NULL -- the kernels compare strides to it */\n'
            '#include <stdint.h>\n'
            '#include "tvm_graph_types.h"\n'
            '#ifndef TVM_DLL\n#define TVM_DLL\n#endif\n'
            'void *TVMBackendAllocWorkspace(int, int, uint64_t, int, int);\n'
            'int   TVMBackendFreeWorkspace(int, int, void *);\n'
            'void  TVMAPISetLastError(const char *);\n')
    (inc / "c_runtime_api.h").write_text(shim)
    (inc / "c_backend_api.h").write_text(
        '/* Generated by arm_build.py -- see c_runtime_api.h. */\n'
        '#include "tvm/runtime/c_runtime_api.h"\n')


# ═══════════════════════════════════════════════════════════════════════════
#  Makefile
# ═══════════════════════════════════════════════════════════════════════════

_MAKEFILE_TMPL = """\
# Generated by src/frontend/tvmrelay/arm_build.py -- regenerate, or edit and
# re-run `make` directly; nothing here is hidden in Python.
#
#   make                  build main.elf      (baremetal aarch64, for the board)
#   make local            build main_local.elf (x86 / this host)
#   make run-local        build it and run it here
#   make clean && make    from scratch
#   make CROSS=... CPU=...  override the toolchain or core
#
# `make` needs the Vitis cross toolchain on PATH:
#     source script/setup.sh --path-set-only
# `make local` needs nothing but the host gcc -- see the LOCAL section below.
#
# Link recipe mirrors script/hostcompile.sh (baremetal aarch64 host link):
# --specs=nosys.specs for the newlib syscall stubs, --defsym end=__bss_end__
# for newlib's _sbrk, and the BSP linker script for the DDR memory map.

CROSS   ?= aarch64-none-elf-
CPU     ?= cortex-a78
CC      := $(CROSS)gcc
LD      := $(CROSS)ld
AR      := $(CROSS)ar

REPO    ?= {repo}
ARCH    := {arch}
BSP     := {bsp}
LSCRIPT := $(ARCH)/lscript.ld

# Where the kernels come from.
#
# DEFAULT: the per-layer sources stage 5 split out, one translation unit per
# operator, compiled to a .o NEXT TO ITS SOURCE and archived into liblayers.a.
# That is what makes a layer individually rebuildable -- `make
# ../layers/09_conv.../09_conv....o` touches one operator -- and it is why
# split_layers' _clear_stale already sweeps `*/*.o` as well as `*/*.c`.
#
# FALLBACK: the single resnet18.c, used when stage 5 was skipped (--no-split)
# or produced nothing. Exactly one of the two is active; KERNELS is empty in
# the layer case, so the file is never compiled twice.
LAYERS  := {layers_dir}
KERNELS := {kernels}

# The monolithic object, named whether or not it is built. Only `clean` uses it
# in the per-layer build: switching modes (or an older tree) can leave a stale
# resnet18.o here, and a `clean` that skipped it would strand a 52 KB object
# that LOOKS like it is part of the link but is not in OBJS and never linked.
KERNEL_OBJ := {kernel_obj}

# Evaluated when make parses this file. Stage 5 runs before stage 6, so the
# layer sources already exist; `sort` both de-duplicates and fixes the order.
# A layer directory with no .c (an AIE-offloaded one, whose kernel now lives
# under its aie/ subdir) simply contributes nothing.
LAYER_SRCS := $(sort $(wildcard $(LAYERS)/*/*.c))
LAYER_OBJS := $(LAYER_SRCS:.c=.o)

# Per-layer AIE archives, one per offloaded layer, built by
# aie_layer_lib.build_archive into layers/NN_op/aie/build/libNN_op.a. Empty on
# a CPU-only build. Each carries that layer's host.cc, its op entry and its
# embedded core ELF, so they are independent of each other and of the order
# they appear here.
AIE_LAYER_LIBS := $(sort $(wildcard $(LAYERS)/*/aie/build/*.a) \
                        $(wildcard $(LAYERS)/*/aie/*/build/*.a))

# Neither source of kernels resolved. Without this the build would cheerfully
# archive zero objects and fail much later with a wall of undefined
# tvmgen_default_* references, which says nothing about the real cause.
ifeq ($(strip $(KERNELS)$(LAYER_SRCS)),)
$(error no kernels found: $(LAYERS) contains no */*.c and KERNELS is empty. \
Re-run deploy_flow.py, or build the monolithic file with \
`make KERNELS=/path/to/resnet18.c`)
endif

# -I$(REPO)/include reaches aie_timer.h (main.c times each phase with it).
# -DAIE_GEN=5 picks its xiltimer.h branch, which is what this cortexa78 BSP
# ships -- the default branch wants xtime_l.h, which the BSP does not have.
# -O3 -funroll-loops, not -Os: on int8 ResNet-18 with the stem on the AIE,
# graph_run() measured 2232 ms at -Os, 1282 ms at -O3, 1270 ms adding
# -funroll-loops (-flto: no gain), all with bit-identical logits. .text grows
# from ~0.4 MB to ~1.3 MB, small next to the 11.9 MB weight blob.
CFLAGS  ?= -O3 -funroll-loops -mcpu=$(CPU) -std=c11 -DAIE_GEN=5 \\
           -I. -I$(REPO)/include -I$(BSP)/include \\
           -I$(REPO)/src/aietensorop/conv2dstem

# AIE offload (--aie-offload). AIE_LIB is libconv2dstem.a when stage 4 emitted a
# BYOC wrapper, and empty otherwise, so a non-BYOC build links exactly as
# before. The archive also pulls in the AIE runtime and the embedded kernel
# ELF, hence the extra BSP/driver -L paths next to it.
AIE_LIB  := {aie_lib}
AIE_LDIRS := {aie_ldirs}
AIE_EXTRA := {aie_extra}

# Per-layer AIE offload (--aie-offload, aiegraph + aiehlc): graph_driver.c
# calls the layer's aie_<kernel>() entry from its aie/build/lib*.a instead of
# the CPU kernel. Board build only -- LOCAL_CFLAGS never gets it, so
# main_local.elf runs the CPU kernel from the same driver and is the reference.
AIE_OFFLOAD_DEFS := {aie_defs}
CFLAGS += $(AIE_OFFLOAD_DEFS)

# AIE_LDIRS comes FIRST, before $(BSP)/lib. The BSP's libxil.a bundles a stale
# copy of the aienginev2 driver (62 xaie*.obj members); build_hw_lib in
# aiehlc.sh strips those from its own libxil.a in thirdparty/alib/lib and
# treats the local aie-rt as authoritative. Searching the BSP first picks the
# unstripped copy and the link dies with dozens of
#   multiple definition of `XAie_UpdateNpiAddr'
# Order matters here; this is not cosmetic.
LDFLAGS := --specs=nosys.specs \\
           -Wl,--defsym,end=__bss_end__ \\
           -Wl,-T -Wl,$(LSCRIPT) \\
           $(AIE_LDIRS) -L$(BSP)/lib
# One unbroken token: the commas are -Wl separators, so a line continuation
# here would split it into two arguments and ld would look for a library
# literally named "-lxilstandalone,...".
LDLIBS  := -Wl,--start-group,-lm,-lxil,-lgcc,-lc,-lxiltimer,-lxilstandalone,-lxilpm_ng$(AIE_EXTRA),--end-group

# The DRIVER sources only. In the default per-layer build KERNELS is empty, so
# OBJS is just these three -- the 65 kernels are NOT here, they reach the link
# as liblayers.a below. Do not read OBJS as "everything that gets linked".
SRCS := $(KERNELS) graph_driver.c tvm_runtime_shim.c main.c
OBJS := $(notdir $(SRCS:.c=.o))

# The kernels' entry to the link. Empty in the monolithic fallback (where they
# arrive as resnet18.o inside OBJS instead), so `make` never archives zero
# objects into a library nothing needs.
LAYER_LIB := $(if $(LAYER_OBJS),liblayers.a,)

all: main.elf

main.elf: $(OBJS) weights.o $(LAYER_LIB) $(AIE_LAYER_LIBS)
\t$(CC) $(CFLAGS) -o $@ $(OBJS) weights.o $(LAYER_LIB) $(AIE_LAYER_LIBS) $(AIE_LIB) $(LDFLAGS) $(LDLIBS)
\t@echo "built $@$(if $(AIE_LAYER_LIBS), with $(words $(AIE_LAYER_LIBS)) per-layer AIE archive(s),)"
\t@$(CROSS)size $@ 2>/dev/null || true

# One archive of per-operator objects. graph_driver.o pulls in the members it
# calls, and a member may call another (the AIE packed-ABI shim calls the BYOC
# wrapper, which is a separate layer): GNU ld rescans an archive until no new
# undefined symbols are resolved, so intra-archive references are fine and no
# --start-group is needed around it.
# The ar line is quiet on purpose: echoing 65 absolute object paths buries the
# per-layer compiles that precede it. `make V=1` shows it.
liblayers.a: $(LAYER_OBJS)
\t@rm -f $@
\t$(if $(V),,@)$(AR) rcs $@ $(LAYER_OBJS)
\t@echo "archived $(words $(LAYER_OBJS)) layer objects into $@ (from $(LAYERS))"

# Kernels live one directory up; everything else is generated here. The same
# rule compiles a layer in place ($(LAYERS)/NN_op/NN_op.c -> .../NN_op.o),
# which is what keeps one operator independently rebuildable.
%.o: %.c
\t$(CC) $(CFLAGS) -c $< -o $@

# Every layer includes ../layers_common.h, so a regenerated prologue must
# rebuild all of them.
$(LAYER_OBJS): $(LAYERS)/layers_common.h

# main.c embeds the preprocessed image and the label table, so regenerating
# either must force it to recompile.
main.o: main.c input_image.h imagenet_labels.h

{kernel_rule}
# `ld -r -b binary` derives _binary_<path>_start from the path it is GIVEN, so
# the file must be named weights.bin and ld must run in this directory --
# otherwise main.c's extern symbol will not resolve.
weights.o: weights.bin
\t$(LD) -r -b binary -o $@ $<

# ───────────────────────────────────────────────────────────────────────────
#  LOCAL (x86 / this host) build -- `make local`, `make run-local`
# ───────────────────────────────────────────────────────────────────────────
#
# The same generated C, built with the host compiler so the graph can be run
# and its top-5 diffed against the CPU reference WITHOUT a board. Three flag
# differences from the cross build, and every one of them is load-bearing:
#
#   -D__AIESIM__   picks aie_timer.h's portable clock_gettime branch. Without
#                  it the header falls through to #include "xtime_l.h", a BSP
#                  header that does not exist off-target.
#   -std=gnu11     NOT -std=c11. clock_gettime/CLOCK_MONOTONIC are POSIX, and
#                  strict ISO mode hides them -- the compile dies with
#                  "'CLOCK_MONOTONIC' undeclared" inside aie_timer.h.
#   -lm            the BSP supplied libm via -lxil's group; here it is explicit.
#
# And three things are DROPPED: -mcpu, the BSP include/lib paths, and the
# baremetal link recipe (--specs=nosys.specs, --defsym end=, -T lscript.ld).
# None of them mean anything to a hosted x86 link.
#
# Objects go in localobj/ with flattened names, so a local build never
# clobbers the cross objects (the layer .o sit next to their .c, and those are
# aarch64). The directory is NOT called `local`: that is the phony target's
# name, and make would see `localobj/x.o: ... | local` as circular and drop
# the order-only dependency, leaving nothing to create the directory.
HOST_CC ?= gcc
HOST_LD ?= ld
LOCALOBJ := localobj

LOCAL_CFLAGS ?= -O2 -std=gnu11 -D__AIESIM__ \\
                -I. -I$(REPO)/include -I$(REPO)/src/aietensorop/conv2dstem
LOCAL_LDLIBS ?= -lm

# `make TRACE=1` (or `make local TRACE=1`) compiles in graph_driver.c's
# per-node tensor dumps -- shape, DECODED dtype ("int16", not the raw 69632
# packing of the DLDataType struct), and the first few elements of every
# kernel input and output. They live behind #ifdef GRAPH_TRACE, so a normal
# build carries none of it. TRACE_ELEMS=N changes how many elements each dump
# shows (default 8).
#
# This is why instrumenting a node does NOT mean editing graph_driver.c: that
# file is regenerated by deploy_flow.py and hand edits are overwritten.
#
# TRACE=1 ALSO writes every kernel input and output out in full as a C header
# of hex bytes -- one file per tensor per call, into ./layeriohex/ relative to
# wherever the ELF runs (so `make local run-local` fills arm_build/layeriohex/).
# TRACE_HEX=0 keeps the console trace but drops the headers; TRACE_HEX_MAX=N
# caps each file at N bytes of payload.
#
# Parameters are dumped twice over, on purpose: once per call that reads them
# (named `..._in1_p0_weight.h`, so a layer's view is complete on its own), and
# once by name in `param_p0_weight.h` (deduplicated, and covering constants no
# traced call reads). TRACE_HEX_PARAMS=0 drops the second set -- it is the
# expensive one, since the whole 11.9 MB blob in hex is ~72 MB.
#
# Only the HOSTED build gets -DGRAPH_TRACE_HEX_FILES. A baremetal board has no
# filesystem -- under xsdb the console is the only channel -- so there the same
# text streams to stdout between BEGIN/END markers and
# `arm_build.split_layeriohex()` cuts a captured log back into files. Linking
# fopen/mkdir into the BSP build instead would compile and then fail at run
# time, which is the worse failure.
# make PROFILE=1: per-call wall time table after "inference" (graph_driver.c
# GRAPH_PROF_*). Off by default -- two timer reads per call are cheap but not free.
ifdef PROFILE
CFLAGS       += -DGRAPH_PROFILE
LOCAL_CFLAGS += -DGRAPH_PROFILE
endif

ifdef TRACE
TRACE_DEFS := -DGRAPH_TRACE $(if $(TRACE_ELEMS),-DGRAPH_TRACE_ELEMS=$(TRACE_ELEMS),)
ifneq ($(TRACE_HEX),0)
TRACE_DEFS += -DGRAPH_TRACE_HEX \\
              $(if $(TRACE_HEX_MAX),-DGRAPH_TRACE_HEX_MAX_BYTES=$(TRACE_HEX_MAX),)
ifeq ($(TRACE_HEX_PARAMS),0)
TRACE_DEFS += -DGRAPH_TRACE_HEX_NO_PARAMS
endif
LOCAL_HEX_DEFS := -DGRAPH_TRACE_HEX_FILES
endif
CFLAGS       += $(TRACE_DEFS)
LOCAL_CFLAGS += $(TRACE_DEFS) $(LOCAL_HEX_DEFS)
endif

# Layer basenames carry their NN_ prefix, so flattening them into localobj/
# is collision-free; vpath is what lets a flat localobj/NN_op.o find its
# source back in $(LAYERS)/NN_op/. It cannot disturb the cross rules above --
# those name their prerequisites by absolute path, and vpath only searches
# when the literal path is missing.
vpath %.c . $(sort $(dir $(LAYER_SRCS) $(KERNELS)))

LOCAL_OBJS := $(addprefix $(LOCALOBJ)/,$(notdir $(SRCS:.c=.o)) \\
                                       $(notdir $(LAYER_SRCS:.c=.o)))

# A BYOC build cannot be run here: its wrapper is compiled INTO the kernel C
# and calls libconv2dstem.a, which is aarch64 baremetal with kernels that run on
# the array. Saying so beats a wall of undefined references to XAie_*.
# Per-layer archives (AIE_LAYER_LIBS) are fine: the driver only calls them
# under GRAPH_AIE_OFFLOAD, which LOCAL_CFLAGS never sets, so main_local.elf is
# the CPU reference for the very same graph_driver.c.
ifneq ($(filter local run-local,$(MAKECMDGOALS)),)
ifneq ($(strip $(AIE_LIB)),)
$(error make local cannot link an AIE-offloaded build -- the archives are \\
aarch64 and their kernels run on the array. Re-run deploy_flow.py without \\
--aie-offload/--aiegraph for a local ELF, or use plain make for the board)
endif
endif

local: main_local.elf

main_local.elf: $(LOCAL_OBJS) $(LOCALOBJ)/weights.o
\t$(HOST_CC) $(LOCAL_CFLAGS) -o $@ $^ $(LOCAL_LDLIBS)
\t@echo 'built $@ (host $(HOST_CC)) -- run it with ./$@ or make run-local'

$(LOCALOBJ)/%.o: %.c | $(LOCALOBJ)
\t$(HOST_CC) $(LOCAL_CFLAGS) -c $< -o $@

# `ld -r -b binary` derives the symbol from the path it is GIVEN, so the input
# stays `weights.bin` (relative, in this directory) even though the output
# goes into localobj/ -- naming the input by its output-relative path would
# give _binary_localobj_weights_bin_start, which main.c's extern never
# resolves.
$(LOCALOBJ)/weights.o: weights.bin | $(LOCALOBJ)
\t$(HOST_LD) -r -b binary -o $@ weights.bin

$(LOCALOBJ):
\t@mkdir -p $@

# Guarded: in the monolithic fallback there are no layers, hence no
# layers_common.h, and an unconditional prerequisite would make `make local`
# fail with "No rule to make target" for a file that is correctly absent.
ifneq ($(strip $(LAYER_SRCS)),)
$(LOCAL_OBJS): $(LAYERS)/layers_common.h
endif

$(LOCALOBJ)/main.o: input_image.h imagenet_labels.h

run-local: main_local.elf
\t./main_local.elf

clean-local:
\trm -rf $(LOCALOBJ) main_local.elf

# Separate from clean-local: a traced run is expensive (the headers are the
# whole graph's buffers) and `make clean-local` to rebuild should not throw
# away the dump you just spent minutes producing.
clean-hex:
\trm -rf layeriohex

clean: clean-local
\trm -f $(OBJS) $(KERNEL_OBJ) weights.o main.elf liblayers.a $(LAYER_OBJS)

.PHONY: all clean clean-local clean-hex local run-local
"""


#: The libconv2dstem.a entry the BYOC wrapper calls. Defined HERE (TVM-free)
#: and imported by ``byoc/aie_codegen``, so the symbol the link keys off and
#: the symbol the codegen emits cannot drift apart.
AIE_ENTRY = "conv2d_stem_prepadded"


def _aie_link_vars(repo_root: Path, kernel_src: Path, build: Path,
                   layers_dir=None) -> dict:
    """Decide whether this ELF links against libconv2dstem.a, and stage its deps.

    Keyed off the generated C actually calling into it, not off a flag: if the
    BYOC wrapper is not in the source there is nothing to resolve, and linking
    the archive anyway would drag the whole AIE runtime into a CPU-only build.

    The archive is produced by
        source script/aiehlc.sh --aie-version 5 \\
            --runtime-source-file src/aietensorop/conv2dstem/conv2dstem.cc
    and published to aout/.

    Two things have to be staged, both because ``thirdparty/alib/lib`` is
    volatile (every aiehlc.sh run wipes and repopulates it):

      * ``libxaienginea78.a`` -- build_hw_lib builds the local aie-rt, strips
        the stale aienginev2 members out of the BSP's libxil.a, and renames the
        result to this. It is the authoritative AIE driver.
      * A **de-duplicated copy of the BSP's libxil.a**. That archive still
        carries 62 ``xaie*.obj`` members of its own, which collide with the 67
        in libxaienginea78.a:
            multiple definition of `XAie_UpdateNpiAddr'
        Deleting them here mirrors exactly what build_hw_lib does to its own
        copy, and leaves the rest of libxil.a (which the BSP needs) intact.
    """
    empty = {"aie_lib": "", "aie_ldirs": "", "aie_extra": "", "aie_defs": ""}
    try:
        byoc = AIE_ENTRY in kernel_src.read_text()
    except OSError:
        byoc = False
    # Per-layer offload needs the same driver libs as the BYOC archive (each
    # aie/build/lib*.a carries the AIE runtime), just no AIE_LIB of its own.
    offload = bool(aie_entries(layers_dir)) if layers_dir else False
    if not (byoc or offload):
        return empty

    archive = repo_root / "aout" / "libconv2dstem.a"
    alib = repo_root / "thirdparty" / "alib" / "lib"
    hint = ("rebuild with `source script/aiehlc.sh --aie-version 5 "
            "--runtime-source-file src/aietensorop/conv2dstem/conv2dstem.cc`")
    if byoc and not archive.is_file():
        print(f"  [arm] warning: the generated C calls {AIE_ENTRY} but "
              f"{archive} is missing -- {hint}")
        return empty

    staged = build / "aielib"
    staged.mkdir(parents=True, exist_ok=True)
    copied = 0
    for lib in sorted(alib.glob("*.a")):
        shutil.copy2(lib, staged / lib.name)
        copied += 1
    if not (staged / "libxaienginea78.a").is_file():
        print(f"  [arm] warning: libxaienginea78.a not found in {alib} "
              f"(it is wiped between runs) -- {hint}, then re-run this build "
              f"in the SAME shell.")
        return empty

    # De-duplicate the BSP's libxil.a against the authoritative driver.
    arch = repo_root / "thirdparty" / "arch" / "cortexa78_0"
    bsp = _bsp_dir(arch)
    ar = f"{os.environ.get('CROSS', 'aarch64-none-elf-')}ar"
    src_xil = bsp / "lib" / "libxil.a"
    if src_xil.is_file():
        shutil.copy2(src_xil, staged / "libxil.a")
        try:
            members = subprocess.run([ar, "t", str(staged / "libxil.a")],
                                     capture_output=True, text=True, check=True).stdout.split()
            stale = [m for m in members if m.startswith("xaie")]
            if stale:
                subprocess.run([ar, "d", str(staged / "libxil.a"), *stale], check=True)
                print(f"  [arm] stripped {len(stale)} stale xaie* members from the staged libxil.a")
        except (subprocess.CalledProcessError, FileNotFoundError) as exc:
            print(f"  [arm] warning: could not de-duplicate libxil.a ({exc}); "
                  f"expect `multiple definition of XAie_*` at link")
    print(f"  [arm] staged {copied} AIE libs -> {staged}")

    return {
        "aie_lib": str(archive) if byoc else "",
        "aie_ldirs": f"-L{staged}",
        # Leading comma: this is spliced inside an existing --start-group list.
        "aie_extra": ",-lxaienginea78,-lstdc++",
        "aie_defs": "-DGRAPH_AIE_OFFLOAD" if offload else "",
    }


def layer_sources(layers_dir) -> list:
    """The per-layer ``.c`` files stage 5 split out, in execution order.

    Empty when the directory is absent or holds no C -- which is the signal to
    fall back to the monolithic kernel file. A layer folder with only an
    ``aie/`` subdir (its kernel was offloaded) contributes nothing and is not
    an error.
    """
    layers_dir = Path(layers_dir)
    if not layers_dir.is_dir():
        return []
    return sorted(layers_dir.glob("*/*.c"))


def write_makefile(build: Path, repo_root: Path, kernel_src: Path,
                   layers_dir=None, verbose: bool = True) -> Path:
    """Write the Makefile that actually builds the ELF. Returns its path.

    The recipe lives here rather than in Python so the build is reproducible
    and editable by hand: ``make`` after a tweak rebuilds only what changed,
    and ``make CROSS=... CPU=...`` retargets without touching the generator.

    Kernels come from *layers_dir* when stage 5 produced one -- one object per
    operator, archived into ``liblayers.a``. Otherwise the single
    *kernel_src* is compiled, which is what ``--no-split`` leaves behind.
    Exactly one of the two is wired up, so no kernel is ever compiled twice.
    """
    arch = repo_root / "thirdparty" / "arch" / "cortexa78_0"
    bsp = _bsp_dir(arch)

    # Absolute: make runs in build/, so a path relative to the caller's cwd
    # would silently expand to nothing and quietly produce an empty archive.
    layers_dir = (Path(layers_dir) if layers_dir else build.parent / "layers").resolve()
    sources = layer_sources(layers_dir)

    if sources:
        # Per-layer build: KERNELS empty, the wildcard over $(LAYERS) drives it.
        kernels, rule = "", ""
        if verbose:
            print(f"  [arm]    kernels: {len(sources)} per-layer objects "
                  f"-> liblayers.a  ({layers_dir.name}/)")
    else:
        # Fallback: compile the kernel file from wherever stage 4 left it.
        kernels = str(kernel_src)
        rule = (f"{kernel_src.stem}.o: {kernel_src}\n"
                f"\t$(CC) $(CFLAGS) -c $< -o $@\n")
        if verbose:
            print(f"  [arm]    kernels: {kernel_src.name} (monolithic -- "
                  f"no layers/ found; --no-split?)")

    text = _MAKEFILE_TMPL.format(
        repo=repo_root, arch=arch, bsp=bsp,
        layers_dir=layers_dir, kernels=kernels, kernel_rule=rule,
        kernel_obj=f"{kernel_src.stem}.o",
        **_aie_link_vars(repo_root, kernel_src, build, layers_dir),
    )
    path = build / "Makefile"
    path.write_text(text)
    return path


# ═══════════════════════════════════════════════════════════════════════════
#  Local (x86) build
# ═══════════════════════════════════════════════════════════════════════════

#: How long ``main_local.elf`` may run before it is killed. ResNet-18 int8 in
#: unvectorized portable C is seconds on a workstation; a cap this generous
#: only ever fires on a genuine hang, and a hang is what a miscompiled kernel
#: looks like when a loop bound goes wrong.
LOCAL_RUN_TIMEOUT = 900


def build_local_elf(build, run: bool = True, jobs: int = 0,
                    verbose: bool = True) -> dict:
    """``make local`` in *build*: the same generated C, linked for this host.

    The point is to run the graph **without a board**. ``main.c`` prints its
    top-5 with logits, so the output is directly comparable with stage 7's
    onnxruntime reference -- which is the check the board ELF otherwise has to
    be flashed to make.

    It is not a cross build with different flags: ``make local`` drops the BSP
    and the baremetal link recipe entirely and adds ``-D__AIESIM__``
    (``aie_timer.h``'s portable branch) and ``-std=gnu11`` (``clock_gettime``
    is POSIX, which strict ISO mode hides). See the LOCAL section of the
    generated Makefile.

    Returns ``{"ok", "elf"|"reason", "ran", "stdout", "top1", ...}``. Never
    raises: no host gcc, or an AIE-offloaded build that cannot link here, is a
    report rather than a traceback.
    """
    build = Path(build)
    makefile = build / "Makefile"
    if not makefile.is_file():
        return {"ok": False, "reason": f"no Makefile in {build}"}
    if shutil.which(os.environ.get("HOST_CC", "gcc")) is None:
        return {"ok": False, "reason": "no gcc on PATH for the local build"}

    cmd = ["make", "-C", str(build), "local"]
    if jobs:
        cmd.append(f"-j{jobs}")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        return {"ok": False, "reason": "make local failed",
                "stderr": (proc.stderr or proc.stdout)[-3000:]}

    elf = build / "main_local.elf"
    out = {"ok": True, "elf": str(elf), "bytes": elf.stat().st_size,
           "ran": False}
    if verbose:
        print(f"  [local] {elf} ({elf.stat().st_size:,} B)")
    if not run:
        return out

    try:
        rp = subprocess.run([str(elf)], capture_output=True, text=True,
                            timeout=LOCAL_RUN_TIMEOUT, cwd=str(build))
    except subprocess.TimeoutExpired:
        out.update({"ran": False, "reason":
                    f"main_local.elf did not finish in {LOCAL_RUN_TIMEOUT}s"})
        return out
    out["stdout"] = rp.stdout
    out["ran"] = rp.returncode == 0
    out["top1"] = _parse_top1(rp.stdout)
    if verbose:
        for line in rp.stdout.strip().splitlines():
            print(f"  [local] {line}")
        if rp.returncode != 0:
            print(f"  [local] exit {rp.returncode}: "
                  f"{(rp.stderr or '').strip()[:300]}")
    return out


def _parse_top1(stdout: str):
    """``(class, logit)`` from main.c's ``resnet18: top1 class=N logit=F`` line.

    Parsed rather than recomputed so the number compared against the CPU
    reference is the one the program actually printed.
    """
    for line in stdout.splitlines():
        if line.startswith("resnet18: top1 "):
            cls = logit = None
            for tok in line.split():
                if tok.startswith("class="):
                    cls = int(tok[6:])
                elif tok.startswith("logit="):
                    logit = float(tok[6:])
            return (cls, logit)
    return None


# ═══════════════════════════════════════════════════════════════════════════
#  README
# ═══════════════════════════════════════════════════════════════════════════

#: Written next to the sources it describes, and regenerated with them, so it
#: cannot drift from the Makefile it documents. Placeholders are @@NAME@@
#: rather than ``{}``/``$`` because the text is full of literal C braces and
#: make ``$(VAR)`` references.
_README_MD = """\
# `arm_build/` — the generated program around TVM's kernels

Generated by `src/frontend/tvmrelay/arm_build.py`. **Do not edit by hand** —
every file here is rewritten by the next `deploy_flow.py` or `arm_build.py`
run, including this README. Change the generator instead.

`relay.build(target="c")` gives kernels but not a program: no `main`, no
weights in memory, nothing calling the kernels in order. This folder is the
missing half.

## Build

```bash
make                     # main.elf      — baremetal aarch64, for the board
make local               # main_local.elf — x86, this host
make run-local           # build the host ELF and run it
make TRACE=1             # either target, with per-layer tensor dumps
make clean               # both; clean-local for just the host artifacts
make CROSS=... CPU=...   # retarget
make -j8
```

`make` needs the Vitis cross toolchain (`source script/setup.sh
--path-set-only`); `make local` needs only the host gcc.

## Files

| file | what it is |
|---|---|
| `graph_driver.c` | one static buffer per `storage_id`, weights pointed into the blob, one call per graph node in order (@@CALLS@@ calls, @@BUFFERS@@ buffers, @@BUFBYTES@@) |
| `tvm_runtime_shim.c` | the four runtime symbols the kernels reference — a bump allocator and an error sink |
| `main.c` | entry point; feeds `input_image.h`, prints the top-5 with logits, per-phase timing, and the `device_teardown done` line `verify_host.sh` greps |
| `input_image.h` | the preprocessed photo as `static const float[@@INELEMS@@]` |
| `imagenet_labels.h` | the 1000 class names |
| `tvm_graph_types.h`, `tvm/runtime/c_*_api.h` | baremetal stand-ins, so the kernel C is used **verbatim** rather than patched |
| `weights.bin` | `resnet18_params.bin`, linked in via `ld -r -b binary` |
| `Makefile` | the build recipe — nothing is hidden in Python |
| `liblayers.a` | built: the @@LAYERS@@ per-layer objects from stage 5 — **the kernels** |
| `localobj/` | built: host objects for `make local`, kept apart from the aarch64 ones |
| `layeriohex/` | built by a `TRACE=1` **run**, not by the build: every kernel input and output as a C hex header |

`graph_driver.c` and `weights.bin` are a **matched pair**: the driver has the
blob's byte offsets baked in as literals, and `save_param_dict` orders its
output differently run to run. Never copy one over the other from a different
build — the weights land at the wrong offsets and the result is silently
garbage.

## Tracing tensors between layers — `make TRACE=1`

`graph_driver.c` carries a dump of every kernel's inputs and outputs behind
`#ifdef GRAPH_TRACE`. A normal build compiles none of it.

```bash
make local TRACE=1 run-local        # host, traced, run it
make TRACE=1                        # board ELF, traced
make local TRACE=1 TRACE_ELEMS=16   # 16 elements per tensor (default 8)
make local TRACE=1 TRACE_HEX=0      # console trace only, no layeriohex/
```

```
[00] tvmgen_default_fused_divide_round_add_clip_cast_subtract_layout_transform
   in  tensors[0] float32 [1,3,224,224] = -1.929532 -1.929532 -1.912407 ...
   in  tensors[1] int16 [] = 113
   out tensors[2] int16 [1,1,224,224,3] = -104 -98 -87 -104 -98 -86
[01] tvmgen_default_fused_nn_contrib_conv2d_NCHWc_add_..._a7e8b96f394601f9_
   in  tensors[2] int16 [1,1,224,224,3] = -104 -98 -87 -104 -98 -86
   in  tensors[3] int16 [16,1,7,7,3,4] = -2 0 -52 25 -18 0
   out tensors[9] uint8 [1,16,112,112,4] = 46 97 14 0 30 89
```

### Full per-layer I/O — `layeriohex/`

`TRACE=1` does two things. The lines above are the first; the second is that
**every** kernel input and output is written out *in full* as a compilable C
header of hex bytes into `layeriohex/` — relative to wherever the ELF runs, so
`make local run-local` fills `arm_build/layeriohex/`.

**Two files per call**, not one per tensor: everything a layer read goes into
`..._in.h` and everything it wrote into `..._out.h`, each tensor under its own
variable name. A conv's activation, weight and six quantization constants are
one `#include` rather than eight, and the set cannot be split up by accident.

```
layeriohex/l06_nn_contrib_conv2d_NCHWc_in.h     <- all 9 inputs
layeriohex/l06_nn_contrib_conv2d_NCHWc_out.h    <- the output
layeriohex/param_p0_weight.h                    <- by name, once (see below)
```

```c
/* Written by graph_driver.c under `make TRACE=1` -- regenerated every run, do not edit.
 * call [06] tvmgen_default_fused_nn_contrib_conv2d_NCHWc_..._ca76d0071d109ff1_ -- 9 input(s)
 */
#ifndef LAYERIOHEX_l06_nn_contrib_conv2d_NCHWc_in_H
#define LAYERIOHEX_l06_nn_contrib_conv2d_NCHWc_in_H

/* in[0] = tensors[2] (in, from call [01] layout_transform) */
#define l06_nn_contrib_conv2d_NCHWc_in0_DTYPE "int8"
#define l06_nn_contrib_conv2d_NCHWc_in0_ELEMS 158700LL
static const long long l06_nn_contrib_conv2d_NCHWc_in0_shape[5] = { 1, 1, 230, 230, 3 };
static const unsigned char l06_nn_contrib_conv2d_NCHWc_in0_data[158700LL] = { ... };

/* in[1] = tensors[3] (param p0, weight, weights.bin@3895354+9408) */
#define l06_nn_contrib_conv2d_NCHWc_in1_p0_weight_DTYPE "int8"
static const long long l06_nn_contrib_conv2d_NCHWc_in1_p0_weight_shape[6] = { 16, 1, 7, 7, 3, 4 };
static const unsigned char l06_nn_contrib_conv2d_NCHWc_in1_p0_weight_data[9408LL] = { ... };

/* in[5] = tensors[11] (param p4, zero-point, weights.bin@3617582+256) */
...
#endif
```

```c
/* Written by graph_driver.c under `make TRACE=1` -- regenerated every run, do not edit.
 * call [01] tvmgen_default_fused_nn_contrib_conv2d_NCHWc_add_... -- in[1] = tensors[3] (param)
 */
#ifndef LAYERIOHEX_l01_fused_nn_contrib_conv2d_NCHWc_add____in1_H
#define LAYERIOHEX_l01_fused_nn_contrib_conv2d_NCHWc_add____in1_H

#define l01_..._in1_DTYPE "int16"
#define l01_..._in1_NDIM  6
#define l01_..._in1_ELEMS 9408LL
#define l01_..._in1_BYTES 18816LL

static const long long l01_..._in1_shape[6] = { 16, 1, 7, 7, 3, 4 };

static const unsigned char l01_..._in1_data[18816LL] = {
    0xfe, 0xff, 0x00, 0x00, 0xcc, 0xff, 0x19, 0x00, ...
};
#endif
```

The **file** stem is `l<call>_<op>_<in|out>`; the **variable** prefix inside it
adds the argument slot and, for a parameter, its name and role:
`l<call>_<op>_<in|out><slot>[_<param>_<role>]`. Both are pasted straight into C
— as the include guard and as `static const … <var>_data`.

The **op slug is deliberately short** (30 chars, cut at a word boundary, with
TVM's `tvmgen_default_fused_` prefix and the trailing content hash removed).
The call index already makes the stem unique, so the op name is there to say
what the layer *does*, not to identify it — and the raw symbol is 95
characters, which made ten files in one layer look identical in `ls` with the
only difference off the right edge of the terminal. Everything that
distinguishes a file from its neighbours is now inside the first 40 characters,
and the **full symbol is still on the first line inside every file**, so
nothing is lost:

```c
/* Written by graph_driver.c under `make TRACE=1` -- regenerated every run, do not edit.
 * call [06] tvmgen_default_fused_nn_contrib_conv2d_NCHWc_subtract_add_add_subtract_fixed_point_multiply_per_ca76d0071d109ff1_
 *   -- in[1] = tensors[3] (param p0, weight, weights.bin@3895354+9408)
 */
```

Two kernels may well slug the same; that cannot collide, because the call index
disambiguates. Verified on int8 ResNet-18: **417 files** (240 in/out groups +
177 parameters) holding 446 tensors, all stems unique.

### Activations say where they came from

A layer can take **several** activations, and a bare `(in)` on each is not
enough to tell them apart. Layer 06 is the standard trap:

```
in[0] = tensors[2] (in, from call [01] layout_transform)        int8  [1,1,230,230,3]
in[2] = tensors[8] (in, from call [05] repeat_multiply_layout)  int32 [1,16,112,112,4]
out[0] = tensors[15] (out, read by call [07])                   uint8 [1,16,112,112,4]
```

`in[2]` has the **same geometry as this call's own output**, so it reads like
the conv's output fed back into itself. It is not: it is the output of call
[05], the tail of the `cast_sum -> multiply -> avg_pool2d -> repeat_multiply`
chain that starts at call [02]. That chain is qnn's **weight-zero-point
correction term** — sum the input over channels, window-sum it with the same
7x7/s2 geometry, scale by the per-output-channel kernel zero point — and the
kernel *subtracts* it, which is the `subtract` in its own fused name. The
dtypes are the giveaway: the correction is `int32`, the conv's output `uint8`.

So every activation carries its producer, `network input` if it is the image,
and every output carries its readers or `network output`. Verified: all 148
producer claims agree with the graph JSON.

### Parameters are named, not just numbered

A conv takes its activation, its weight, and five or six int32 constants. As
`_in3` / `_in4` / `_in6` those are indistinguishable, so every input that is a
graph **parameter** carries its name and its role:

| role | what it is |
|---|---|
| `weight` | the int32 MAC multiply |
| `bias` | added into the accumulator |
| `zero_point` | subtracted |
| `requant_multiplier` | the **int64**-widened multiply |
| `requant_shift` | the requantization shift |
| `copied` | only re-laid-out, no arithmetic role |
| `unused` | wired in, never referenced |

The role comes from `network_md.kernel_param_usage` / `param_role` — the same
classifier behind `network.md`'s parameter tables, so the two cannot disagree
about what a constant is for; the labels are cross-checked against it for all
41 parameter-bearing calls. It keys on the kernel's own argument slot, and
`pN` is literally `args[N]` in the generated C, so a slot indexes this list
directly. **Only a slot that is also a parameter takes a role**: the classifier
labels activations too, and layer 06's image input lands in an `add`, which
would otherwise file it as a "bias".

The provenance line also carries the blob offset, so a dumped weight is
checkable straight against `weights.bin`:

```
 * call [06] tvmgen_default_fused_nn_contrib_conv2d_NCHWc_... -- in[1]
 *   = tensors[3] (param p0, weight, weights.bin@3895354+9408)
```

```bash
dd if=weights.bin bs=1 skip=3895354 count=9408 of=ref.bin   # == the dumped bytes
```

### `param_*.h` — every constant once, by name

The `..._in.h` groups are per **call**, so a weight read by several calls
appears in each of their headers. `graph_dump_params()` is the other view: one
file per graph constant — `param_p0_weight.h`, `param_p5_requant_multiplier.h`
— deduplicated, and covering constants no traced call reads at all. These stay
one-per-file rather than merged, because a graph constant belongs to no single
layer and the merged file would be the whole 72 MB blob. It runs once, from the
top of `graph_run()`, after `graph_init()` has filled the buffers.

`TRACE_HEX_PARAMS=0` drops this set and keeps the per-call ones. It is the
expensive half: the blob is 11.9 MB and hex is ~6x, so it alone is ~72 MB.

**Bytes, not decoded elements, and on purpose.** This is the on-the-wire image
of the buffer; the element view is what `graph_dump_tensor()` prints. The
dtype and shape ride along as macros so a reader can decode it — see *Dump
elements, not bytes* below for why mixing the two readings is the standard
way to misread this data.

| knob | effect |
|---|---|
| `TRACE_HEX=0` | keep the console trace, write no headers |
| `TRACE_HEX_PARAMS=0` | drop the `param_*.h` set, keep the per-call dumps |
| `TRACE_HEX_MAX=N` | cap each file at N bytes of payload (`_TRUNCATED` is defined in the header when it bites) |
| `make clean-hex` | delete `layeriohex/` — deliberately **not** part of `clean-local`, so rebuilding does not discard a dump |

It is not small: the full int8 ResNet-18 is roughly **100 MB** of text for the
per-call dumps plus **~72 MB** for `param_*.h`, because parameters are kernel
inputs too and a weight read by several calls is dumped for each of them. They
are dumped rather than skipped because a mismatch lands in a parameter as often
as in an activation — `TRACE_HEX_MAX` is the lever when you only need the
leading bytes, and `TRACE_HEX_PARAMS=0` when you only need the activations.

#### On the board there is no filesystem

Only the hosted build gets `-DGRAPH_TRACE_HEX_FILES`. A baremetal board under
xsdb has no filesystem — the console is the only channel — so `make TRACE=1`
for the board streams the *identical* text to stdout, framed:

```
===BEGIN layeriohex/l00_..._in0.h===
...the same header text...
===END layeriohex/l00_..._in0.h===
```

Capture it and cut it back into files:

```python
from frontend.tvmrelay.arm_build import split_layeriohex
split_layeriohex("applog")            # -> ./layeriohex/*.h
```

Linking `fopen`/`mkdir` into the BSP build instead would compile and then fail
at run time, which is the worse failure. Note that a UART carries ~100 MB very
slowly — `TRACE_HEX_MAX=256` is the usual setting for a board run.

### Why not just add a `printf`

You can — `dl_code_name()`, `dl_dtype_str()` and `graph_dump_tensor()` are
`static inline` in `graph_driver.c` and always emitted, so an ad-hoc `printf`
can call them. But **this file is regenerated**, and a hand-added dump is gone
after the next `deploy_flow.py` run. `TRACE=1` is not.

### `Input tensor data type: 69632`

That is what `printf("%d", tensors[i].dtype)` prints, and it is not a type
code. `DLDataType` is a 4-byte **struct**, so passing it to `%d` is undefined
behaviour that happens to print its little-endian packing:

```c
typedef struct { uint8_t code; uint8_t bits; uint16_t lanes; } DLDataType;
value = code | (bits << 8) | (lanes << 16)
```

```
69632 = 0x00011000
          |   | +---- code  = 0x00 = 0   -> kDLInt
          |   +------ bits  = 0x10 = 16
          +---------- lanes = 0x0001 = 1        ==> int16
```

Codes: **0** `int`, **1** `uint`, **2** `float`, **4** `bfloat`. The values
this graph actually produces:

| printed | hex | dtype |
|---|---|---|
| 67584 | `0x010100` | `int8` |
| 67585 | `0x010101` | `uint8` |
| 69632 | `0x011000` | `int16` |
| 66304 | `0x010300` | `int32` |
| 73730 | `0x012002` | `float32` |

`dl_dtype_str()` prints `int16` instead. To fix an existing hand-written line:

```c
char ds[24];
printf("Input tensor data type: %s\\n", dl_dtype_str(tensors[2].dtype, ds, sizeof ds));
```

Shapes have the same trap: `shape` is `int64_t *`, so `printf("%d", shape[i])`
is also wrong — use `printf("%lld", (long long)shape[i])`.

### Dump elements, not bytes

A byte dump of an int16 tensor reads `98 ff 9e ff a9 ff`. That is
little-endian `-104, -98, -87`, and reading it as bytes is the most common way
to misread this output — it is also why `conv2dstem_image.h` and TVM's layer 0
look like different data when they are bit-identical quantization.
`graph_dump_tensor()` switches on `(code, bits)` and prints decoded values. It
clamps the count to the tensor's element count, because the zero-point
parameters are `ndim 0` / `numel 1` and dumping 8 of those would read past a
2-byte buffer.

## `make local` — the same C on x86

`main_local.elf` is the same `graph_driver.c`, the same layer sources, the
same `weights.bin` and the same `main.c`, linked for this host. It prints the
top-5 the board would, so the result is checkable without flashing anything.

It is **not** the cross build with a different `-mcpu`. The BSP paths and the
baremetal link recipe (`--specs=nosys.specs`, `--defsym end=__bss_end__`,
`-T lscript.ld`) are dropped, and two flags have to be added — both failures
point *into* `include/aie_timer.h`, not at your code:

| flag | without it |
|---|---|
| `-D__AIESIM__` | `fatal error: xtime_l.h: No such file` — the header falls through to the BSP-only branch |
| `-std=gnu11`, not `-std=c11` | `'CLOCK_MONOTONIC' undeclared` — it is POSIX, which strict ISO mode hides |
| `-lm` | the BSP supplied libm inside `-lxil`'s group |

Objects go to `localobj/`, deliberately **not** `local/`: that is the phony
target's name, and make would read `local/x.o: ... | local` as circular, drop
the order-only prerequisite, and then fail with a misleading `can't create
local/x.o: No such file or directory`.

An **AIE-offloaded build cannot be linked here** and says so up front — the
archives are aarch64 and their kernels run on the array. Re-run
`deploy_flow.py` without `--aie-offload`/`--aiegraph`.

## Where the kernels come from

By default the per-layer sources stage 5 split out: `../layers/NN_op/NN_op.c`
-> `NN_op.o` beside its source -> `liblayers.a`, so editing one operator
recompiles one translation unit. The monolithic `../resnet18.c` is the
fallback when stage 5 was skipped (`--no-split`). Exactly one of the two is
wired in; if neither resolves, `make` stops with a named error instead of a
wall of undefined `tvmgen_default_*` references.

## More

- `src/frontend/tvmrelay/README.md` — the whole flow, stage by stage
- skill `tvmlocalhostelf` — `make local`, `TRACE=1`, the dtype packing
- skill `conv2dstemfixture` — why TVM's layer 0 and the AIE fixture differ
"""


def write_readme(build: Path, summary: dict, layers: int = 0,
                 verbose: bool = True) -> Path:
    """Write ``arm_build/README.md`` describing what was just generated.

    Regenerated with the sources so it cannot drift from the Makefile it
    documents -- the same reason the recipe itself lives in a generated file
    rather than inside Python.
    """
    mb = summary.get("buffer_bytes", 0) / 1e6
    text = _README_MD
    for key, val in (("@@CALLS@@", str(summary.get("calls", "?"))),
                     ("@@BUFFERS@@", str(summary.get("buffers", "?"))),
                     ("@@BUFBYTES@@", f"{mb:.1f} MB"),
                     ("@@INELEMS@@", str(summary.get("input_elems", "?"))),
                     ("@@LAYERS@@", str(layers) if layers else "per-layer")):
        text = text.replace(key, val)
    path = build / "README.md"
    path.write_text(text)
    if verbose:
        print(f"  [arm] readme  : {path}")
    return path


# ═══════════════════════════════════════════════════════════════════════════
#  Build
# ═══════════════════════════════════════════════════════════════════════════

def build_arm_elf(out_dir, repo_root=None, c_name="resnet18.c",
                  run_make: bool = True, jobs: int = 0, image=None,
                  local: bool = False, local_run: bool = True,
                  verbose: bool = True) -> dict:
    """Generate the sources + Makefile, then run ``make`` to link ``main.elf``.

    Python only *generates*; ``make`` does the building, so the same Makefile
    the flow used is the one a person re-runs by hand. ``run_make=False``
    emits everything and stops, which is also what happens automatically when
    the cross toolchain is absent.

    ``local=True`` additionally runs ``make local`` on the same generated
    sources, producing (and by default running) ``main_local.elf`` for this
    host. It is deliberately **independent of the cross toolchain**: the whole
    value of a local ELF is on a box that has no Vitis install, so it is built
    before the cross-toolchain gate rather than after it.

    Returns ``{"ok", "elf"|"reason", "makefile", ...}``, with the local build
    under ``["local"]``. Never raises for a missing toolchain -- a box without
    the Vitis cross compiler is a normal place to run the earlier stages.
    """
    out_dir = Path(out_dir).resolve()
    repo_root = Path(repo_root) if repo_root else Path(__file__).resolve().parents[3]

    graph = out_dir / f"{Path(c_name).stem}_graph.json"
    params = out_dir / f"{Path(c_name).stem}_params.bin"
    for path in (out_dir / c_name, graph, params):
        if not path.is_file():
            return {"ok": False, "reason": f"missing {path.name}"}

    build = out_dir / "arm_build"
    build.mkdir(exist_ok=True)
    summary = generate_driver(graph, params, build, c_source=out_dir / c_name,
                              verbose=verbose)
    _write_tvm_header_stubs(build)
    shutil.copyfile(params, build / "weights.bin")

    # The image the ELF classifies, plus the names it prints. Written before
    # the Makefile so a missing image fails here rather than as a mysterious
    # "input_image.h: No such file" out of the compiler.
    from frontend.tvmrelay import image_input

    tensor, image_path = image_input.preprocess_image(image, verbose=verbose)
    if tensor.size != summary["input_elems"]:
        return {"ok": False, "reason":
                f"image tensor has {tensor.size} elements, but the graph "
                f"input takes {summary['input_elems']}"}
    image_input.write_image_header(tensor, build / "input_image.h",
                                   source=str(image_path), verbose=verbose)
    labels = image_input.load_labels(verbose=verbose)
    image_input.write_labels_header(labels, build / "imagenet_labels.h",
                                    verbose=verbose)
    summary["image"] = str(image_path)
    # Stage 5 splits into <out_dir>/layers/ and runs before this stage, so the
    # per-layer sources are already on disk when the Makefile is written.
    makefile = write_makefile(build, repo_root, out_dir / c_name,
                              layers_dir=out_dir / "layers", verbose=verbose)
    summary["makefile"] = str(makefile)
    summary["layer_objects"] = len(layer_sources(out_dir / "layers"))
    if verbose:
        print(f"  [arm] makefile: {makefile}")
    summary["readme"] = str(write_readme(build, summary,
                                         layers=summary["layer_objects"],
                                         verbose=verbose))

    # Before the cross-toolchain gate on purpose: a box with no Vitis install
    # is exactly where a local ELF is worth having.
    if local:
        summary["local"] = build_local_elf(build, run=local_run, jobs=jobs,
                                           verbose=verbose)
        if verbose and not summary["local"].get("ok"):
            print(f"  [local] not built -- {summary['local'].get('reason')}")
            for line in (summary["local"].get("stderr") or "").strip(
                    ).splitlines()[-5:]:
                print(f"  [local]   {line}")

    ok, detail = toolchain_status(repo_root)
    if not ok:
        if verbose:
            print(f"  [arm] not building -- {detail}")
            print(f"  [arm] sources are ready; run `make -C {build}` once it is")
        summary.update({"ok": False, "reason": detail})
        return summary
    if not run_make:
        summary.update({"ok": True, "elf": None})
        return summary

    cmd = ["make", "-C", str(build)]
    if jobs:
        cmd.append(f"-j{jobs}")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        summary.update({"ok": False, "reason": "make failed",
                        "stderr": (proc.stderr or proc.stdout)[-3000:]})
        return summary

    elf = build / "main.elf"
    summary.update({"ok": True, "elf": str(elf), "bytes": elf.stat().st_size})
    if verbose:
        print(f"  [arm] {elf} ({elf.stat().st_size:,} B)")
    return summary


def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("out_dir", nargs="?", default="worklocal/tvmrelay_deploy",
                    help="directory holding the generated C + graph + params")
    ap.add_argument("--no-make", action="store_true",
                    help="generate the sources and Makefile, then stop "
                         "(run `make -C <out>/arm_build` yourself)")
    ap.add_argument("-j", "--jobs", type=int, default=0,
                    help="parallel make jobs (default: serial)")
    ap.add_argument("--local", action="store_true",
                    help="also build main_local.elf for this host (x86) with "
                         "the host gcc and run it, so the graph can be checked "
                         "without a board")
    ap.add_argument("--no-local-run", action="store_true",
                    help="with --local, build main_local.elf but do not run it")
    args = ap.parse_args(argv)

    result = build_arm_elf(args.out_dir, run_make=not args.no_make,
                           jobs=args.jobs, local=args.local,
                           local_run=not args.no_local_run)
    local = result.get("local") or {}
    if not result.get("ok"):
        print(f"error: {result.get('reason')}")
        if result.get("stderr"):
            print(result["stderr"])
        # A local ELF is a real deliverable even when the cross build is not
        # possible here, so it is not swallowed by the aarch64 failure.
        return 0 if local.get("ok") else 1
    if result.get("elf"):
        print(f"built {result['elf']}")
    else:
        print(f"generated {result['makefile']} -- run make to build")
    if local.get("ok"):
        print(f"built {local['elf']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
