###############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
"""``ag.dnn_build(app)``: an aiednn application -> one ELF, kernels from the library."""

from __future__ import annotations

import json
import os
import re
import shutil
import copy
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

from . import _env
from .errors import AiegraphError
from .kernel_lib import kernel_lib

__all__ = ["DnnBuild", "dnn_build", "TARGETS"]

#: aiednn API call -> (aiednn_primitive_kind_t, kernel-library op family)
PRIMITIVES = {"aiednn_matmul_primitive_create": ("aiednn_matmul", "dense"),
              "aiednn_matmul_forward_create": ("aiednn_matmul", "dense"),
              "aiednn_inner_product_forward_create": ("aiednn_inner_product", "dense"),
              "aiednn_convolution_primitive_create": ("aiednn_convolution", "conv2d"),
              "aiednn_convolution_forward_create": ("aiednn_convolution", "conv2d"),
              "aiednn_deconvolution_forward_create": ("aiednn_deconvolution", "conv2d"),
              "aiednn_matmul_forward_create_attr": ("aiednn_matmul", "dense"),
              "aiednn_inner_product_forward_create_attr": ("aiednn_inner_product", "dense"),
              "aiednn_convolution_forward_create_attr": ("aiednn_convolution", "conv2d"),
              "aiednn_deconvolution_forward_create_attr": ("aiednn_deconvolution", "conv2d"),
              # 8-bit per-element ops: the GEMM ELF's table mode (lut_blk_s8, indexed as conv2d)
              "aiednn_eltwise_forward_create": ("aiednn_eltwise", "conv2d"),
              "aiednn_prelu_forward_create": ("aiednn_prelu", "conv2d"),
              "aiednn_binary_create": ("aiednn_binary", "conv2d"),
              "aiednn_batch_normalization_forward_inference_create": ("aiednn_batch_normalization", "conv2d"),
              "aiednn_batch_normalization_forward_inference_create_attr": ("aiednn_batch_normalization", "conv2d"),
              "aiednn_reorder_create": ("aiednn_reorder", "conv2d"),
              # 8-bit reductions / pooling: the pool mode (pool_blk_s8s32)
              "aiednn_reduction_create": ("aiednn_reduction", "conv2d"),
              "aiednn_pooling_forward_create": ("aiednn_pooling", "conv2d"),
              "aiednn_pooling_forward_create_nd": ("aiednn_pooling", "conv2d"),
              # the gemm+ / gemmx composites: every one runs its matrix products on the GEMM block
              "aiednn_convolution_forward_create_nd": ("aiednn_convolution", "conv2d"),
              "aiednn_deconvolution_forward_create_nd": ("aiednn_deconvolution", "conv2d"),
              "aiednn_convolution_backward_data_create": ("aiednn_convolution", "conv2d"),
              "aiednn_convolution_backward_weights_create": ("aiednn_convolution", "conv2d"),
              "aiednn_deconvolution_backward_data_create": ("aiednn_deconvolution", "conv2d"),
              "aiednn_deconvolution_backward_weights_create": ("aiednn_deconvolution", "conv2d"),
              "aiednn_inner_product_backward_data_create": ("aiednn_inner_product", "dense"),
              "aiednn_inner_product_backward_weights_create": ("aiednn_inner_product", "dense"),
              "aiednn_grouped_matmul_create": ("aiednn_matmul", "dense"),
              "aiednn_rnn_forward_create": ("aiednn_rnn", "dense"),
              "aiednn_rnn_backward_create": ("aiednn_rnn", "dense"),
              # sdpa's AIE modes (sdpa_blk, bf16x3) live in gemm_uk.cc, the conv2d family's source
              "aiednn_sdpa_forward_create": ("aiednn_sdpa", "conv2d"),
              "aiednn_sdpa_forward_create_attr": ("aiednn_sdpa", "conv2d"),
              # linear attention: the chunked form's f32 GEMMs run on the GEMM ELF's bf16x3 (gemm_uk.cc)
              "aiednn_linear_attention_create": ("aiednn_linear_attention", "conv2d"),
              # f32 per-row ops on the FVEC mode / 8-bit ones on STEPS (the GEMM ELF)
              "aiednn_softmax_forward_create": ("aiednn_softmax", "conv2d"),
              "aiednn_layer_normalization_forward_inference_create": ("aiednn_layer_normalization", "conv2d"),
              "aiednn_layer_normalization_forward_inference_create_attr": ("aiednn_layer_normalization", "conv2d"),
              "aiednn_group_normalization_forward_inference_create": ("aiednn_group_normalization", "conv2d")}
#: any other aiednn_*_create (incl. suffixed: _create_nd, _create_attr) runs on ref: (no kernel family needed)
_ANY_CREATE = re.compile(r"\baiednn_(?!stream_|graph_)\w+_create(?:_\w+)?\s*\(")
#: kernel-library op family -> the aiednn primitive kind its AIEK_ENTRY implements
KIND_OF = {"dense": "aiednn_matmul", "conv2d": "aiednn_convolution"}
#: entries whose aiednn primitive kind is not their op family's (the GEMM ELF's overlay modes;
#: the runtime still checks each one's signature, e.g. is_pool_block)
ENTRY_KIND = {"pool_blk_s8s32": "aiednn_pooling", "lut_blk_s8": "aiednn_eltwise",
              "elt_blk_s8s32": "aiednn_binary", "sdpa_blk_s8s32": "aiednn_sdpa", "bf16_blk_f32": "aiednn_matmul",
              "fvec_blk_f32": "aiednn_eltwise", "wq_blk_s8f32": "aiednn_matmul",
              "fpool_blk_f32": "aiednn_pooling"}   # fvec: the runtime binds it to eltwise / binary / softmax / norms
_DT = {"int8": "aiednn_s8", "uint8": "aiednn_u8", "int32": "aiednn_s32", "float32": "aiednn_f32"}
TARGETS = {"vek280": {"aie_version": "2", "runner": "vek280_mlir_aie/run_vek280.py",
                      # one resident kernel: loaded + launched once, later launches only re-arm the
                      # shim BDs (--persistent; the GEMM ELF is a megakernel whose mode rides in a
                      # descriptor); --warm-launch covers any non-persistent relaunch
                      "aiehlc_args": ("--warm-launch", "--persistent"),
                      # the runtime's im2col / epilogue / packing loops run on the A72 in host.cc
                      "host_opt": "-O3",
                      "stage": "/proj/xsjsswstaff/{user}/vek280_mlir_aie"},
           # VEK385 / AIE2PS: an A78 host; script/test/appvek385.py copies the ELF itself (no stage dir)
           "vek385": {"aie_version": "5", "runner": "script/test/appvek385.py",
                      "aiehlc_args": ("--warm-launch", "--persistent"),
                      "host_opt": "-O3"},
           "x86": {}}

#: aiehlc args that need a kernel feature: --persistent re-arms only the shim BDs, so the core must
#: program its own DMA (persist_transform.py refuses a host that programs core tiles)
_ARG_NEEDS = {"--persistent": "#pragma KERNELCONFIGOFFLOAD"}


@dataclass
class DnnBuild:
    app: Path
    target: str
    out: Path
    kernels: list                       # LibEntry
    ok: bool = False
    elf: Path = None
    reason: str = ""
    log: Path = None
    run_ok: bool = None
    run_log: Path = None
    run_lines: list = field(default_factory=list)
    modes: tuple = None                 # megakernel modes compiled (None: the source's defaults, all)
    segments: list = None               # several megakernels: [{name, first, last, modes, cost}] (_plan_megakernels)
    mk_of_op: list = None               # op index -> segment (the implementation list's table)
    mk_need: dict = None                # modes="library": entry -> mask of the library ELFs that hold its modes

    def __repr__(self) -> str:
        ks = ", ".join(f"aie:{e.name} ({e.contract.source.name})" for e in self.kernels) or "none (ref: only)"
        lines = [f"aiednn build {self.app.name} -> {self.target}: "
                 + (f"{self.elf}" if self.ok else f"FAILED -- {self.reason}"),
                 f"  implementations: {ks}, then ref:",
                 f"  out: {self.out}" + (f"  (log {self.log.name})" if self.log else "")]
        if self.run_ok is not None:
            lines.append(f"  run: {'PASS' if self.run_ok else 'FAIL'}" +
                         (f"  (log {self.run_log})" if self.run_log else ""))
            lines += [f"    {l}" for l in self.run_lines[-12:]]
        return "\n".join(lines)


def needed_families(text: str) -> dict:
    """aiednn primitive kind -> kernel-library op family"""
    fams = {kind: fam for call, (kind, fam) in PRIMITIVES.items() if re.search(rf"\b{call}\s*\(", text)}
    if re.search(r"\bwei_scales\b", text) and "aiednn_matmul" in fams:
        fams["aiednn_matmul_wq"] = "wq"
    return fams


def creates_primitives(text: str) -> bool:
    return bool(_ANY_CREATE.search(text))


def is_gemm_block(spec) -> bool:
    """An s8 x s8 -> s32 GEMM block"""
    a = spec.args
    return (len(a) == 3 and [x.kind for x in a] == ["in", "param", "out"]
            and [x.dtype for x in a] == ["int8", "int8", "int32"] and all(len(x.dims) == 2 for x in a))


#: entries that only serve the runtime's composites (no primitive of their own): they ride along
#: with their source but do not count when choosing a source
HELPER_ENTRIES = {"gemm_q_s8s8", "gemm_kc_s8s32", "dwq_blk_s8s8", "conv_kc_s8s32", "conv_q_s8s8",
                  "sdpa_blk_s8s32", "bf16_blk_f32", "fvec_blk_f32"}   # sdpa rides with the GEMM source


def covers(entry) -> dict:
    """op family -> 'direct' | 'composite' for what this kernel can serve."""
    if entry.contract.spec.entry == "wq_blk_s8f32":   # grouped weight-only quant (wq:)
        return {"wq": "direct"}
    if entry.contract.spec.entry in HELPER_ENTRIES:
        return {}
    if is_gemm_block(entry.contract.spec):   # serves both only through blocked: / im2col+
        return {"dense": "composite", "conv2d": "composite"}
    return {entry.contract.op: "direct"}


def _pick_kernels(lib, families: set, kernels, board: str) -> tuple:
    """Kernels for *board* that serve the most of *families*, from ONE source (one aiehlc library per ELF)."""
    why = []
    cand = []
    for e in lib:
        if kernels and e.name not in ({kernels} if isinstance(kernels, str) else set(kernels)):
            continue
        if not (e.runnable and e.verified):
            why.append(f"{e.name}: not runnable / not verified")
        elif board not in e.contract.spec.targets:
            why.append(f"{e.name}: AIEK_SPEC target {' '.join(e.contract.spec.targets) or '(none)'}, "
                       f"not {board}")
        elif e.contract.spec.entry in HELPER_ENTRIES:
            cand.append(e)                          # rides with its source, never decides it
        elif not set(covers(e)) & families:
            why.append(f"{e.name}: {e.contract.op} serves none of {', '.join(sorted(families))}")
        else:
            cand.append(e)
    if kernels:
        missing = ({kernels} if isinstance(kernels, str) else set(kernels)) - {e.name for e in lib}
        if missing:
            raise AiegraphError("dnn_build", f"no kernel named {', '.join(sorted(missing))} in the library",
                                hint="print(ag.kernel_lib(root)) lists them")
    by_src = {}
    for e in cand:
        by_src.setdefault(e.contract.source, []).append(e)
    scored = []
    for src, es in list(by_src.items()):
        if all(e.contract.spec.entry in HELPER_ENTRIES for e in es):
            del by_src[src]                         # a source of helpers only serves nothing
            continue
        cov = {}
        for e in es:
            for fam, how in covers(e).items():
                if fam in families and cov.get(fam) != "direct":
                    cov[fam] = how
        # third key: how many of the source's entries serve the families, so the general GEMM library
        # (gemm_uk.cc) beats a one-shape kernel (VEK385's conv2dstem_uk.cc) on a conv2d-only app
        nserve = sum(1 for e in es if set(covers(e)) & families)
        scored.append(((len(cov), sum(v == "direct" for v in cov.values()), nserve), src, es, cov))
    scored = [t for t in scored if t[3]]           # a source must serve at least one family
    if not scored:
        return [], why
    scored.sort(key=lambda t: t[0], reverse=True)
    for score, src, es, cov in scored:
        why.append(f"{src.name}: serves {', '.join(f'{f} ({h})' for f, h in sorted(cov.items()))} "
                   f"score {score}")
    if len(scored) > 1 and scored[0][0] == scored[1][0]:
        raise AiegraphError("dnn_build", "kernels from several aiehlc sources serve the app equally: "
                            + ", ".join(sorted(t[1].name for t in scored if t[0] == scored[0][0])),
                            hint="one aiehlc library per ELF in this tree (doc/tutorial_userkernel.md "
                                 "§5); pass kernels=[...] to pick one source")
    return scored[0][2], why


def _dims(arg, syms: dict) -> list:
    out = []
    for d in arg.dims:
        if isinstance(d, int):
            out.append(str(d))
        elif d == "*":
            out.append("AIEDNN_DIM_ANY")
        else:
            out.append(f"AIEDNN_DIM_SYM({syms.setdefault(d, len(syms))})")
    return out


def _mesh_dims(spec, mesh) -> dict:
    """{(arg name, dim): value} for the spec's ``mesh`` dims on *mesh* = (rows, cols)"""
    if not mesh:
        return {}
    S = _env.aiekernel_pure("spec")
    return {(n, d): S.mesh_dim(c, u, *mesh) for n, d, c, u in getattr(spec, "mesh", ())}


def impl_list_source(picked: list, x86: bool, mesh=None, count: bool = False, mk_of_op=None,
                     mk_need=None) -> str:
    """C++ for ``aiednn_impl_list[]``."""
    decls, rows, thunks, needs = [], [], [], []
    for e in picked:
        spec = e.contract.spec
        md = _mesh_dims(spec, mesh)
        if len(spec.args) > 8 or any(a.dtype not in _DT for a in spec.args):
            raise AiegraphError("dnn_build", f"{e.name}: AIEK_SPEC arguments aiednn cannot describe",
                                hint=f"dtypes {', '.join(_DT)}, at most 8 arguments")
        fn = f"{spec.entry}_ref" if x86 else spec.entry
        decls.append(f'extern "C" int {fn}(const aiek_tensor *args, int num_args);')
        if count:
            k = len(thunks)
            thunks.append(f"static int aiednn_count_{k}(const aiek_tensor *a, int n) "
                          f"{{ aiednn_entry_calls[{k}]++; aiednn_count_op({k}); return {fn}(a, n); }}")
            fn = f"aiednn_count_{k}"
        if mk_need and spec.entry in mk_need:   # load a library megakernel holding the entry's modes
            k = len(needs)
            needs.append(f"static int aiednn_need_{k}(const aiek_tensor *a, int n) "
                         f"{{ if (aiednn_megakernel_need) aiednn_megakernel_need({mk_need[spec.entry]:#x}u); "
                         f"return {fn}(a, n); }}")
            fn = f"aiednn_need_{k}"
        syms, args, fixed = {}, [], 0
        for a in spec.args:
            if any(k[0] == a.name for k in md):
                a = copy.copy(a)
                a.dims = tuple(md.get((a.name, i), d) for i, d in enumerate(a.dims))
            dims = _dims(a, syms)
            fixed += sum(isinstance(d, int) for d in a.dims)
            args.append(f"{{{_DT[a.dtype]}, {len(a.dims)}, {{{', '.join(dims)}}}}}")
        kind = ENTRY_KIND.get(spec.entry, KIND_OF[e.contract.op])
        rows.append(f'    {{"aie:{e.name}", {kind}, {fn}, {len(spec.args)},\n'
                    f'     {{{", ".join(args)}}},\n'
                    f'     {fixed}, "{e.contract.source.name}"}},')
    if not rows:
        rows = ['    {"none", aiednn_matmul, nullptr, 0, {}, 0, ""},   // placeholder: count is 0']
    count = len(picked)
    if thunks:   # call counts per entry, printed at exit: which kernel modes the app really needs
        names = ", ".join(f'"{e.contract.spec.entry}"' for e in picked)
        decls += ["", "#include <stdio.h>", "#include <stdlib.h>", f"static long aiednn_entry_calls[{count}];",
                  # per (entry, op): which ops need which entries (the multi-megakernel planner)
                  'extern "C" int aiednn_cur_op;',
                  f"static long aiednn_entry_op_calls[{count}][AIEDNN_USAGE_MAX_OPS];",
                  "static void aiednn_count_op(int k) {",
                  "    if (aiednn_cur_op >= 0 && aiednn_cur_op < AIEDNN_USAGE_MAX_OPS)",
                  "        aiednn_entry_op_calls[k][aiednn_cur_op]++;",
                  "}", *thunks,
                  f"static const char *const aiednn_entry_names[{count}] = {{{names}}};",
                  # readers for a ctypes host (dnn_viz: which entries -- opcodes -- each op used)
                  f'extern "C" int aiednn_entry_count(void) {{ return {count}; }}',
                  f'extern "C" const char *aiednn_entry_name(int i) {{ return i >= 0 && i < {count} ? aiednn_entry_names[i] : ""; }}',
                  f'extern "C" long aiednn_entry_calls_get(int i) {{ return i >= 0 && i < {count} ? aiednn_entry_calls[i] : 0; }}',
                  f'extern "C" void aiednn_entry_calls_reset(void) {{ for (int i = 0; i < {count}; i++) aiednn_entry_calls[i] = 0; }}',
                  "__attribute__((destructor)) static void aiednn_entry_usage(void) {",
                  '    if (!getenv("AIEDNN_ENTRY_USAGE"))',
                  "        return;",
                  f"    for (int i = 0; i < {count}; i++) {{",
                  '        fprintf(stderr, "AIEDNN_ENTRY_USED %s %ld\\n", aiednn_entry_names[i], aiednn_entry_calls[i]);',
                  "        for (int o = 0; o < AIEDNN_USAGE_MAX_OPS; o++)",
                  "            if (aiednn_entry_op_calls[i][o])",
                  '                fprintf(stderr, "AIEDNN_ENTRY_OP %s %d %ld\\n", aiednn_entry_names[i], o, aiednn_entry_op_calls[i][o]);',
                  "    }",
                  "}"]
    if thunks:
        decls.insert(0, "#define AIEDNN_USAGE_MAX_OPS 4096")
    if needs:
        decls += ["", 'extern "C" void aiednn_megakernel_need(unsigned mask) __attribute__((weak));', *needs]
    if mk_of_op:   # several megakernels: op -> segment (aiednn_op_begin -> aiednn_megakernel_select)
        decls += ["", f'extern "C" const int aiednn_mk_nops = {len(mk_of_op)};',
                  'extern "C" const unsigned char aiednn_mk_of_op[] = {' + ", ".join(map(str, mk_of_op)) + "};"]
    return ("// Generated by aiegraph/dnn.py from the kernel library's AIEK_SPECs. Do not edit.\n"
            '#include "aiednn_impl.h"\n\n' + "\n".join(decls) + "\n\n"
            'extern "C" const aiednn_impl_t aiednn_impl_list[] = {\n' + "\n".join(rows) + "\n};\n"
            f'extern "C" const int aiednn_impl_count = {count};\n')


def _entry_source(app: Path, picked: list, impl_cc: Path, mesh=None) -> str:
    root = _env.repo_root()
    lines = ["// Generated by aiegraph/dnn.py: the aiehlc entry for an aiednn app. Do not edit.",
             "// Kernel source(s) first: aiehlc must see the kernel before anything that",
             "// reaches its launch."]
    if mesh:  # the kernel's mesh (gemm.h: HW_ROWS x HW_COLS cores, block and partition follow)
        lines += [f"#define HW_ROWS {mesh[0]}", f"#define HW_COLS {mesh[1]}"]
    # diagnostic defines for the kernel and runtime, e.g. AIEDNN_DEFINES="AIEDNN_KCYC=1" (core cycle stamps)
    for d in os.environ.get("AIEDNN_DEFINES", "").split():
        name, _, val = d.partition("=")
        lines.append(f"#define {name} {val or 1}")
    lines += [f'#include "{s}"' for s in dict.fromkeys(e.contract.source for e in picked)]
    lines += [f'#include "{root / "src" / "aiednn" / "aiednn_rt.cc"}"',
              f'#include "{impl_cc.name}"',
              f'#include "{app}"', ""]
    return "\n".join(lines)


def _copy_headers(dirs, out: Path) -> None:
    """aiehlc.sh copies only the ENTRY dir's ``*.h`` (flat) for the host compile."""
    for d in dict.fromkeys(dirs):
        for h in Path(d).glob("*.h"):
            shutil.copy2(h, out / h.name)


def _vitis_prefix(vitis_settings) -> str:
    vs = vitis_settings or os.environ.get("AIEDNN_VITIS_SETTINGS")
    if vs:
        return f"source {vs} > /dev/null 2>&1 && "
    if os.environ.get("XILINX_VITIS"):
        return ""
    raise AiegraphError("dnn_build", "no Vitis environment",
                        hint="pass vitis_settings=<.../Vitis/settings64.sh>, set AIEDNN_VITIS_SETTINGS, "
                             "or source Vitis first")


#: board apps' heap (_HEAP_SIZE; the BSP default 128 MB is too small for large f32 composites).
#: dnn_build(heap=) overrides; 0 keeps the BSP's.
APP_HEAP_BYTES = 0x20000000


def _build_board(rep: DnnBuild, entry: Path, aiehlc_args, vitis_settings, verbose: bool) -> None:
    root = _env.repo_root()
    cfg = TARGETS[rep.target]
    rep.log = rep.out / "aiehlc.log"
    elf = rep.out / "aout" / "main.elf"
    if elf.exists():
        elf.unlink()
    start = time.time()
    host_opt = f"export AIEHLC_HOST_OPT='{cfg['host_opt']}' && " if cfg.get("host_opt") else ""
    heap = getattr(rep, "heap", APP_HEAP_BYTES)   # the BSP script's default is 128 MB
    ldflags = f"-Wl,--defsym,_HEAP_SIZE={heap:#x}" if heap else ""
    extra_ld = os.environ.get("AIEHLC_HOST_LDFLAGS_EXTRA", "").strip()
    if extra_ld:
        ldflags = (ldflags + " " + extra_ld).strip()
    host_opt += f"export AIEHLC_HOST_LDFLAGS='{ldflags}' && " if ldflags else ""
    host_libs = os.environ.get("AIEHLC_HOST_LIBS", "").strip()
    if host_libs:
        host_opt += f"export AIEHLC_HOST_LIBS='{host_libs}' && "
    srcs = "".join(s.read_text(errors="replace") for s in dict.fromkeys(e.contract.source for e in rep.kernels))
    args = [a for a in cfg.get("aiehlc_args", ()) if _ARG_NEEDS.get(a, "") in srcs]
    cmd = (f"{_vitis_prefix(vitis_settings)}cd {rep.out} && {host_opt}source {root}/script/aiehlc.sh "
           f"--aie-version {cfg['aie_version']} {' '.join((*args, *aiehlc_args))} "
           f"--runtime-source-file {entry}")
    print(f"[aiegraph] dnn_build: aiehlc.sh --aie-version {cfg['aie_version']} in {rep.out} "
          f"(a few minutes; log {rep.log.name})")
    with open(rep.log, "w") as log:
        subprocess.run(["bash", "-c", cmd], stdout=None if verbose else log,
                       stderr=subprocess.STDOUT if not verbose else None)
    build = rep.out / "aout" / "worklocal" / "build"
    # with per-model segments only kernel_mk* reach the cores; a library build has none and every map counts
    pm, st = core_program_memory(build), core_stack(build)
    if "kernel_mk0.map" in pm:
        pm = {m: n for m, n in pm.items() if m.startswith("kernel_mk")}
        st = {m: v for m, v in st.items() if m.startswith("kernel_mk")}
    over = {m: n for m, n in pm.items() if n > CORE_PM_BYTES}
    deep = {m: f for m, (f, size) in st.items() if f > size - CORE_STACK_MARGIN}
    if over:   # links and loads, then the core runs garbage: refuse the ELF (a megakernel's modes outgrew it)
        rep.reason = ("core program memory overflow: " + ", ".join(f"{m} {n} B" for m, n in sorted(over.items())) +
                      f" > {CORE_PM_BYTES} (drop a mode from that megakernel; MODE_PM / LIBRARY_MEGAKERNELS)")
    elif deep:  # the kernel frame runs into the statics below the stack: wrong pixels, nothing reported
        rep.reason = ("core stack overflow: " + ", ".join(f"{m} frame {f} B" for m, f in sorted(deep.items())) +
                      f" (stack minus {CORE_STACK_MARGIN} B for the callers); shrink kernel locals or drop a mode")
    elif elf.is_file() and elf.stat().st_mtime >= start:
        rep.ok, rep.elf = True, elf
    elif any(a.stat().st_mtime >= start for a in (rep.out / "aout").glob("lib*.a")):
        rep.ok = True   # an entry without main(): aiehlc archived a library (dnn_lib.build_library)
    else:
        rep.reason = f"no fresh aout/main.elf; see {rep.log}"


CORE_PM_BYTES = 16384


CORE_STACK_MARGIN = 512   # main + libc frames above the kernel's


def core_stack(build: Path) -> dict:
    """kernel*.map -> (the largest function stack frame, the DM_stack size) in bytes"""
    out = {}
    for m in sorted(Path(build).glob("kernel*.map")):
        t = m.read_text(errors="replace")
        frames = [int(f) for f in re.findall(r"stack frame size = (\d+)", t)]
        size = re.search(r"\(\s*(\d+) items\) : Stack", t)
        if frames and size:
            out[m.name] = (max(frames), int(size.group(1)))
    return out


def core_program_memory(build: Path) -> dict:
    """kernel*.map -> its core program memory total (bytes): the 'PM' section summary chess writes"""
    out = {}
    for m in sorted(Path(build).glob("kernel*.map")):
        t = re.search(r"Section summary for memory 'PM':.*?(\d+)\s+Total", m.read_text(errors="replace"), re.S)
        if t:
            out[m.name] = int(t.group(1))
    return out


def _build_x86(rep: DnnBuild, picked: list, impl_cc: Path, headers=(), mesh=None) -> None:
    root = _env.repo_root()
    # the kernel references' x86 stand-ins for whole-launch entries (n-tile / MMT / dwq) size a launch from
    # the mesh, as the board build does (gemm.h): the same block the impl list carries
    mdefs = [f"-DHW_ROWS={mesh[0]}", f"-DHW_COLS={mesh[1]}"] if mesh else []
    refs = [e.contract.spec.reference for e in picked]
    if any(r is None for r in refs):
        raise AiegraphError("dnn_build", "target x86 needs every kernel's AIEK_REFERENCE",
                            hint="add `reference <file>.c` to the AIEK_SPEC")
    rep.elf = rep.out / f"{rep.app.stem}_x86"
    rep.log = rep.out / "x86_build.log"
    objs = []
    with open(rep.log, "w") as log:
        for r in dict.fromkeys(refs):          # one source can hold several entries' references
            o = rep.out / f"{Path(r).stem}.o"
            subprocess.run(["gcc", "-O2", "-std=gnu11", *mdefs, f"-I{root / 'include'}", f"-I{root / 'src' / 'aietensorop' / 'ops'}",
                            "-c", str(r), "-o", str(o)],
                           check=False, stdout=log, stderr=subprocess.STDOUT)
            objs.append(str(o))
        res = subprocess.run(["g++", "-O2", "-std=gnu++17", f"-I{root / 'include'}", f"-I{root / 'src' / 'aiednn'}",
                              f"-I{root / 'src' / 'aietensorop' / 'ops'}",
                              *[f"-I{Path(h).resolve()}" for h in headers], f"-I{rep.app.parent}",
                              str(root / "src" / "aiednn" / "aiednn_rt.cc"), str(impl_cc), str(rep.app),
                              *objs, "-o", str(rep.elf)], stdout=log, stderr=subprocess.STDOUT)
    rep.ok = res.returncode == 0 and rep.elf.is_file()
    if not rep.ok:
        rep.reason = f"host g++ failed; see {rep.log}"


_USED = re.compile(r"^AIEDNN_ENTRY_USED (\w+) (\d+)$", re.M)


def entry_usage(app: Path, picked: list, out: Path, headers=(), timeout: int = 1800) -> dict:
    """Build *app* for x86 with the board's kernel choice"""
    out.mkdir(parents=True, exist_ok=True)
    rep = DnnBuild(app, "x86", out, picked)
    impl_cc = out / "aiednn_impl_list.cc"
    impl_cc.write_text(impl_list_source(picked, x86=True, count=True))
    _build_x86(rep, picked, impl_cc, headers)
    if not rep.ok:
        raise AiegraphError("dnn_build", f"usage build failed: {rep.reason}")
    res = subprocess.run([str(rep.elf)], capture_output=True, text=True, timeout=timeout, cwd=out,
                         env=dict(os.environ, AIEDNN_ENTRY_USAGE="1"))
    (out / "run.log").write_text(res.stdout + res.stderr)
    used = {m.group(1): int(m.group(2)) for m in _USED.finditer(res.stderr)}
    if res.returncode or not used:
        raise AiegraphError("dnn_build", f"usage run failed (rc {res.returncode}); see {out / 'run.log'}")
    return used


_USED_OP = re.compile(r"^AIEDNN_ENTRY_OP (\w+) (\d+) (\d+)$", re.M)


def usage_by_op(log_text: str) -> dict:
    """{op index: {entry: calls}} from a usage run's AIEDNN_ENTRY_OP lines (aiednn_op_begin)"""
    out = {}
    for m in _USED_OP.finditer(log_text):
        out.setdefault(int(m.group(2)), {})[m.group(1)] = int(m.group(3))
    return out


#: core program memory (bytes) and stack of the GEMM megakernel per optional mode (from kernel.map).
#: Program memory adds per mode; per-mode buffers share `dwb`, so stack takes the largest.
MK_PM_BASE, MK_STACK_BASE = 4464, 10848
MK_PM_LIMIT, MK_STACK_LIMIT = 16384 - 512, 32768 - 1024        # margins: libc / call frames
MODE_PM = {"GEMMQ": 2944, "DW": 1024, "POOL": 3792, "DWQ": 2640, "CONV": 3008,
           "ELTWISE": 960, "LUT": 1344, "STEPS": 1344, "MMT": 2272, "SDPA": 6896, "BF16": 1632,
           "WQ": 6432 - 4464,   # kernel.map of WQ alone + base
           "FPOOL": 8432 - 4464,   # FPOOL (+ ARGMAX for max backward) alone + base
           # entries that cannot fit even alone are left out of a per-app build (_restrict_to_usage)
           "FVEC": 15232 - 4464}  # FVEC alone
MODE_STACK_ADD = {"GEMMQ": 2528, "CONV": 800}                   # locals of the mode's code
MODE_DWB = {"DWQ": 8192, "LUT": 8192, "STEPS": 8192, "CONV": 16384, "MMT": 5120, "FVEC": 7680}
#: cost of one megakernel switch (ELF reload on the next launch)
MK_SWITCH_MS = 0.5
MODE_UNKNOWN = (3000, 8192)


def megakernel_cost(modes) -> tuple:
    """(program memory, stack frame) estimate of a megakernel compiled with *modes*"""
    pm = MK_PM_BASE + sum(MODE_PM.get(m, MODE_UNKNOWN[0]) for m in modes)
    dwb = max([MODE_DWB.get(m, MODE_UNKNOWN[1] if m not in MODE_PM else 0) for m in modes] or [0])
    return pm, MK_STACK_BASE + sum(MODE_STACK_ADD.get(m, 0) for m in modes) + dwb


def plan_segments(op_modes: list) -> list:
    """Split the op sequence into contiguous segments whose mode union fits one core program"""
    def fits(ms):
        pm, st = megakernel_cost(ms)
        return pm <= MK_PM_LIMIT and st <= MK_STACK_LIMIT
    segs, cur, first = [], set(), 0
    for i, ms in enumerate(op_modes):
        if not fits(ms):
            raise AiegraphError("dnn_build", f"op {i} needs modes {sorted(ms)}, which do not fit one core program "
                                f"({megakernel_cost(ms)} vs {MK_PM_LIMIT} B PM / {MK_STACK_LIMIT} B stack)")
        if fits(cur | ms):
            cur |= ms
            continue
        segs.append((first, i - 1, cur))
        cur, first = set(ms), i
    segs.append((first, len(op_modes) - 1, cur))
    return segs


#: The library's fixed megakernel set (modes="library"): the runtime switches between these ELFs on demand
#: (aiednn_megakernel_need); ELF 0 is resident at the first launch. Every set fits MK_PM_LIMIT / MK_STACK_LIMIT.
LIBRARY_MEGAKERNELS = (("cnn", ("GEMMQ", "CONV", "POOL", "DW")),                 # convs, pools
                       ("dwq", ("GEMMQ", "DW", "DWQ", "MMT")),                   # depthwise blocks
                       ("act", ("GEMMQ", "ELTWISE", "LUT", "STEPS", "POOL")),    # per-element ops
                       ("attn", ("GEMMQ", "SDPA", "ELTWISE")),                   # transformer blocks
                       ("float", ("BF16", "GEMMQ", "POOL", "ELTWISE")),          # bf16 / f32 (bf16x3) GEMMs
                       ("fvec", ("FVEC",)),   # f32 pointwise / softmax / norms
                       ("wq", ("GEMMQ", "WQ", "ELTWISE", "FPOOL")))   # grouped weight-only quant
# FVEC fills a core program by itself (the vector-program interpreter, aie_modes_fvec.h): with GEMMQ it overflows


def library_need(entries: dict, sets=LIBRARY_MEGAKERNELS) -> dict:
    """entry -> bitmask of the library ELFs (index in *sets*) that compile all of the entry's modes"""
    need = {}
    for entry, ms in entries.items():
        if not ms:
            continue
        mask = sum(1 << k for k, (_, s) in enumerate(sets) if set(ms) <= set(s))
        if not mask:
            raise AiegraphError("dnn_build", f"{entry}: modes {' '.join(sorted(ms))} are in no library megakernel",
                                hint="add them to a set in dnn.LIBRARY_MEGAKERNELS (within MK_PM_LIMIT)")
        need[entry] = mask
    return need


def modes_header(default: str, on: set) -> str:
    """aie_modes_cfg.h with only the modes in *on* enabled"""
    def sub(m):  # 0 and 1 both ways: an opt-in mode (default 0, e.g. MMT) turns on when the app uses it
        return f"#define AIEK_MODE_{m.group(1)} {1 if m.group(1) in on else 0}"
    text = re.sub(r"#define AIEK_MODE_([A-Z0-9_]+) [01](?=\s)", sub, default)
    return "/* Generated by aiegraph/dnn.py (modes=\"auto\"): the modes this app uses. Do not edit. */\n" + text


def _restrict_to_usage(picked: list, used: dict) -> tuple:
    """(entries kept, modes on): an entry with optional modes stays only if the app called it"""
    keep = [e for e in picked if not e.contract.spec.modes or used.get(e.contract.spec.entry, 0)]
    # an entry whose own modes exceed a core program can run in no megakernel: leave it out (dispatch
    # serves those primitives with the next implementation), as the library build does
    big = [e for e in keep if e.contract.spec.modes and not _fits(e.contract.spec.modes)]
    if big:
        print("[aiegraph] dnn_build: leaves out " + ", ".join(
            f"{e.contract.spec.entry} ({' '.join(e.contract.spec.modes)}: {megakernel_cost(e.contract.spec.modes)[0]} B)"
            for e in big) + f": over {MK_PM_LIMIT} B of core program alone")
        keep = [e for e in keep if e not in big]
    on = sorted({m for e in keep for m in e.contract.spec.modes})
    return keep, on


def _fits(modes) -> bool:
    pm, st = megakernel_cost(modes)
    return pm <= MK_PM_LIMIT and st <= MK_STACK_LIMIT


def dnn_build(app, target: str = "vek280", out=None, root=None, kernels=None, aiehlc_args=(),
              vitis_settings=None, run: bool = False, board: str = None, timeout: int = 120,
              verbose: bool = False, emulate: str = "vek280", headers=(), mesh=None,
              modes: str = "auto", heap: int = APP_HEAP_BYTES) -> DnnBuild:
    """Build an aiednn app for *target* with kernels from the library at *root* (default ``src/aietensorop``)"""
    if target not in TARGETS:
        raise AiegraphError("dnn_build", f"target={target!r}", hint=f"one of {', '.join(TARGETS)}")
    app = Path(app).resolve()
    if not app.is_file():
        raise AiegraphError("dnn_build", f"{app} does not exist")
    repo = _env.repo_root()
    text = app.read_text(errors="replace")
    kinds = needed_families(text)
    if not creates_primitives(text):
        raise AiegraphError("dnn_build", f"{app.name} creates no aiednn primitive",
                            hint="include/aiednn.h: aiednn_<primitive>_create(...)")
    lib = kernel_lib(root or repo / "src" / "aietensorop", quiet=True)
    picked, why = _pick_kernels(lib, set(kinds.values()), kernels, emulate if target == "x86" else target)
    for line in why:
        print(f"[aiegraph] dnn_build: {line}")
    out = Path(out or repo / "worklocal" / "aiednn" / f"{app.stem}_{target}").resolve()
    out.mkdir(parents=True, exist_ok=True)
    rep = DnnBuild(app, target, out, picked)
    impl_cc = out / "aiednn_impl_list.cc"
    mesh = tuple(mesh) if mesh and tuple(mesh) != (4, 4) else None
    for old in out.glob("aie_modes_cfg_*.h"):   # a previous build's variants would be compiled in too
        old.unlink()
    if modes not in ("auto", "all", "library"):
        raise AiegraphError("dnn_build", f"modes={modes!r}", hint='"auto", "all" or "library"')
    if target != "x86" and modes == "auto" and any(e.contract.spec.modes for e in picked):
        picked = _apply_usage(rep, app, picked, out, headers)
    elif target != "x86" and modes == "library":
        _plan_library(rep, picked, lib)
    impl_cc.write_text(impl_list_source(picked, x86=target == "x86", mesh=mesh, mk_of_op=rep.mk_of_op,
                                        mk_need=rep.mk_need))
    if target == "x86":
        _build_x86(rep, picked, impl_cc, headers, mesh)
    else:
        if not picked:
            raise AiegraphError("dnn_build", "no AIE kernel in the library serves what the app creates "
                                "(an aiehlc build needs one)",
                                hint="target='x86' runs it on the ref: implementations; the AIE "
                                     "kernels serve the GEMM family (matmul, inner product, (de)convolution)")
        entry = out / f"{app.stem}_aiednn.cc"
        entry.write_text(_entry_source(app, picked, impl_cc, mesh))
        # the runtime's own headers LAST: an app dir can hold stale copies of them from an
        # earlier build, and a stale aiednn_rt_gemm.h next to a current aiednn_rt.cc fails to compile
        _copy_headers([e.contract.source.parent for e in picked] + [app.parent] + [Path(h) for h in headers]
                      + [repo / "src" / "aietensorop" / "ops", repo / "src" / "aiednn"], out)
        if rep.modes is not None:   # after the copy: shadows the default (all modes) header
            default = (repo / "src" / "aietensorop" / "ops" / "aie_modes_cfg.h").read_text()
            cfg = modes_header(default, set(rep.modes))
            if rep.mk_need:   # library: the host routes to a mode some ELF holds (aiednn_megakernel_need loads it)
                cfg += ("\n#if !defined(__AIENGINE__) /* host: the modes of the library's megakernel set */\n" +
                        "".join(f"#define AIEK_LIB_MODE_{m} 1\n" for m in sorted({m for sg in rep.segments
                                                                                 for m in sg["modes"]})) + "#endif\n")
            (out / "aie_modes_cfg.h").write_text(cfg)
            for k, sg in enumerate(rep.segments or []):   # one core program per segment (multi-ELF build)
                if k or not rep.mk_need:   # library: ELF 0 is the default kernel.o (aie_modes_cfg.h)
                    (out / f"aie_modes_cfg_{k}.h").write_text(modes_header(default, set(sg["modes"])))
            if rep.segments:
                (out / "megakernels.json").write_text(json.dumps(
                    dict(megakernels=[dict(sg, cfg=f"aie_modes_cfg_{k}.h" if k or not rep.mk_need else "aie_modes_cfg.h")
                                 for k, sg in enumerate(rep.segments)],
                         mk_of_op=rep.mk_of_op), indent=1))
        rep.heap = heap
        _build_board(rep, entry, aiehlc_args, vitis_settings, verbose)
    if rep.ok and run:
        run_app(rep, board=board, timeout=timeout)
    return rep


def _apply_usage(rep: DnnBuild, app: Path, picked: list, out: Path, headers) -> list:
    """modes="auto": the entries the app calls on x86 decide the megakernel's modes"""
    try:
        used = entry_usage(app, picked, out / "usage_x86", headers)
    except (AiegraphError, OSError, subprocess.TimeoutExpired) as exc:
        print(f"[aiegraph] dnn_build: modes=auto: {exc}; building every mode")
        return picked
    keep, on = _restrict_to_usage(picked, used)
    allm = sorted({m for e in picked for m in e.contract.spec.modes})
    rep.modes, rep.kernels = tuple(on), keep
    _plan_megakernels(rep, app, keep, out / "usage_x86")
    print(f"[aiegraph] dnn_build: megakernel modes {' '.join(on) or '(GEMM only)'}; dropped "
          f"{' '.join(m for m in allm if m not in on) or 'none'} (app run on x86: "
          + ", ".join(f"{k} {n}" for k, n in sorted(used.items()) if n) + ")")
    return keep


def _plan_library(rep: DnnBuild, picked: list, lib=()) -> None:
    """modes="library": every entry stays and the fixed LIBRARY_MEGAKERNELS are built"""
    sets = library_sets(lib, rep.target)
    held = set().union(*(set(ms) for _, ms in sets))
    out = [e for e in picked if not set(e.contract.spec.modes) <= held]
    if out:
        print("[aiegraph] dnn_build: library leaves out " +
              ", ".join(f"{e.contract.spec.entry} ({' '.join(e.contract.spec.modes)})" for e in out) +
              ": no LIBRARY_MEGAKERNELS set holds its modes")
        picked[:] = [e for e in picked if e not in out]
    rep.mk_need = library_need({e.contract.spec.entry: e.contract.spec.modes for e in picked}, sets)
    rep.modes = sets[0][1]
    rep.segments = [dict(name=n, modes=sorted(ms), cost=list(megakernel_cost(ms))) for n, ms in sets]
    print(f"[aiegraph] dnn_build: library megakernels " +
          "; ".join(f"mk{k} {n}: {' '.join(ms)} ({megakernel_cost(ms)[0]} B PM)"
                    for k, (n, ms) in enumerate(sets)))


def library_sets(lib, board: str) -> tuple:
    """LIBRARY_MEGAKERNELS without the modes only entries for other boards declare"""
    mine = {m for e in lib if board in e.contract.spec.targets for m in e.contract.spec.modes}
    other = {m for e in lib if board not in e.contract.spec.targets for m in e.contract.spec.modes} - mine
    return tuple((n, tuple(m for m in ms if m not in other)) for n, ms in LIBRARY_MEGAKERNELS)


def _plan_megakernels(rep: DnnBuild, app: Path, keep: list, udir: Path) -> None:
    """several megakernels when the app's modes do not fit one core program"""
    log = udir / "run.log"
    by_op = usage_by_op(log.read_text()) if log.is_file() else {}
    m = re.search(r"#define NOPS (\d+)", app.read_text(errors="replace"))
    nops = int(m.group(1)) if m else (max(by_op) + 1 if by_op else 0)
    modes_of = {e.contract.spec.entry: set(e.contract.spec.modes) for e in keep}
    op_modes = [set().union(*[modes_of.get(e, set()) for e, n in by_op.get(i, {}).items() if n]) for i in range(nops)]
    segs = plan_segments(op_modes) if nops else []
    if len(segs) <= 1:   # the app's modes fit one core program: today's single-megakernel build
        rep.segments, rep.mk_of_op = None, None
        return
    # one core program per DISTINCT mode set
    distinct = list(dict.fromkeys(tuple(sorted(ms)) for _, _, ms in segs))
    rep.segments = [dict(name=f"mk{k}", first=min(a for a, _, m in segs if tuple(sorted(m)) == ms),
                         last=max(b for _, b, m in segs if tuple(sorted(m)) == ms), modes=list(ms),
                         cost=list(megakernel_cost(ms))) for k, ms in enumerate(distinct)]
    seg_k = [distinct.index(tuple(sorted(ms))) for _, _, ms in segs]
    rep.mk_of_op = [seg_k[next(j for j, (a, b, _) in enumerate(segs) if a <= i <= b)] for i in range(nops)]
    print(f"[aiegraph] dnn_build: {len(segs)} segments on {len(distinct)} megakernels (modes do not fit one core "
          f"program; ~{(len(segs) - 1 + (len(segs) > 1)) * MK_SWITCH_MS:.1f} ms of switches per run on VEK280): " +
          "; ".join(f"mk{seg_k[j]} ops {a}..{b} {' '.join(sorted(ms)) or 'GEMM'}" for j, (a, b, ms) in enumerate(segs)))


def run_app(rep: DnnBuild, board: str = None, timeout: int = 120) -> DnnBuild:
    """Run a built app: x86 locally, vek280 on *board* via ``run_vek280.py``."""
    repo = _env.repo_root()
    rep.run_log = rep.out / "run.log"
    if rep.target == "x86":
        res = subprocess.run([str(rep.elf)], capture_output=True, text=True, timeout=timeout)
        text = res.stdout + res.stderr
        rep.run_ok = res.returncode == 0 and "PASS:" in text
    else:
        if not board:
            raise AiegraphError("dnn_build", "run on vek280 needs board=", hint="board='galerina13'")
        if rep.elf is None:   # the build archived a library: the entry has no main() aiehlc recognizes
            raise AiegraphError("dnn_build", f"no ELF to run ({rep.app.name} built as a library)",
                                hint="aiehlc recognizes `int main()` -- `int main(void)` makes the app a library")
        user = os.environ.get("USER", "")
        if rep.target == "vek385":
            text = _run_vek385(rep, board, user, timeout)
            rep.run_ok = "PASS:" in text and "FAIL:" not in text
            rep.run_log.write_text(text)
            keep = ("aiednn", "PASS", "FAIL", "MISMATCH", "ERROR", "Exception", "BENCH")
            rep.run_lines = [l for l in text.splitlines() if any(k in l for k in keep)]
            return rep
        host = board if "@" in board else f"{user}@{board}"
        stage = Path(TARGETS[rep.target]["stage"].format(user=user))
        stage.mkdir(parents=True, exist_ok=True)
        staged = stage / f"aiednn_{rep.app.stem}_{host.split('@')[-1]}.elf"   # per board: two boards run at once
        shutil.copy2(rep.elf, staged)
        print(f"[aiegraph] dnn_build: running {staged.name} on {host} (power cycle + JTAG, ~1-2 min)")
        res = subprocess.run(["python3", "-u", str(repo / TARGETS[rep.target]["runner"]), "--host", host,
                              "--timeout", str(timeout), str(staged)],
                             capture_output=True, text=True, errors="replace")
        text = (res.stdout + res.stderr).replace("\r", "")
        rep.run_ok = "PASS:" in text and "FAIL:" not in text
    rep.run_log.write_text(text)
    keep = ("aiednn", "PASS", "FAIL", "MISMATCH", "ERROR", "Exception")
    rep.run_lines = [l for l in text.splitlines() if any(k in l for k in keep)]
    return rep


def _run_vek385(rep: DnnBuild, board: str, user: str, timeout: int) -> str:
    """VEK385: script/test/appvek385.py (JTAG dow + UART capture; it scp's the ELF itself)."""
    repo = _env.repo_root()
    host = board.split("@")[-1]
    env = dict(os.environ, USERNAME=board.split("@")[0] if "@" in board else user, VEK385IP=host,
               AIEHLC_DOWNLOAD_TIMEOUT=str(max(240, timeout)))
    text = ""
    for attempt in range(3):   # the PDI load intermittently fails ("PLM stalled during programming"): retry
        print(f"[aiegraph] dnn_build: running {rep.elf.name} on {host} (power cycle + JTAG, ~2 min)"
              + (f", retry {attempt} after a failed PDI load" if attempt else ""))
        res = subprocess.run(["python3", "-u", str(repo / TARGETS[rep.target]["runner"]), "-y", str(rep.elf)],
                             capture_output=True, text=True, errors="replace", env=env, timeout=timeout + 900)
        text = (res.stdout + res.stderr).replace("\r", "")
        if "PDIRES_FAIL" not in text:
            break
    return text


_OP_LINE = re.compile(r"^OP (\S+) impl=(\S+) sum=(\S+) asum=(\S+)")


def compare_ops_logs(a, b, rtol: float = 1e-5) -> list:
    """Differences between two ``ops_app`` logs (e.g. board vs x86); empty list = they agree."""
    def ops(text):
        return [m.groups() for m in map(_OP_LINE.match, Path(text).read_text(errors="replace")
                                        .replace("\r", "").splitlines()) if m]
    oa, ob = ops(a), ops(b)
    if len(oa) != len(ob):
        return [f"{len(oa)} OP lines vs {len(ob)}"]
    out = []
    for (na, ia, sa, aa), (nb, ib, sb, ab) in zip(oa, ob):
        if (na, ia) != (nb, ib):
            out.append(f"{na} on {ia} vs {nb} on {ib}")
            continue
        for x, y, what in ((float(sa), float(sb), "sum"), (float(aa), float(ab), "asum")):
            if abs(x - y) > rtol * max(abs(x), abs(y), 1.0):
                out.append(f"{na}: {what} {x!r} vs {y!r}")
    return out
