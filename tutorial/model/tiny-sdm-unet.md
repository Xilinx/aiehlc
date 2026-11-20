# Run the tiny-SD UNet on the AIE with aiegraph and generate an orange

This tutorial walks you step by step through deploying the UNet of **bk-sdm-tiny** (a small 256×256 Stable Diffusion model) to a first-generation VEK385 board (palmyra, A78 APU + AIE2PS): the A78 runs the C code TVM generates, and conv, Linear and attention run on the AIE. Then the text encoder and the VAE run on the host while every UNet step runs on the board, and together they generate a picture of an orange.

The whole flow uses only **aiegraph**, the same framework as the [ResNet18](resnet18.md) and [YOLOv26](yolov26.md) tutorials. On the user side you write only `build_unet.py` (listed in full in section 5) and an offload declaration, `unet_offload.py`. All C text (AIE kernels, host glue, the board main) lives in aiegraph and in the templates under [`src/aiednn/`](../../src/aiednn/).

Every command was run on palmyra213 (10.23.224.213); the results are in [Measured results](#measured-results) at the end.

![The orange generated on the board](tiny-sdm-unet-orange.png)

---

## 0. How it works

```
                 ┌──────────────────── build_unet.py (x86, once, ~25 min) ─────────────────────┐
 unet_sim.onnx ─► aiegraph.load ─► m.convert(target="a78") ─► TVM C: graph_driver.c (layer by layer on the A78)
                                      │
           partition_p4a.json ─► unet_offload.offload(): emod.match / offload_group / driver_template
                                      │   156 layers go to the AIE: 48 conv + 72 Linear + 36 attention (flash-attention groups)
                                      │   kernels / host glue come from src/aiednn/{gemm,conv2d,attention,...}
                                      ▼
               emod.build(target="a78", runtime_io=...) ─► main.elf (A78 code + AIE kernels + bf16 weights + .aiehlc_io input section)
                 └──────────────────────────────────────────────────────────────────────────────┘

 Image generation (run_orange.sh):
   prompt ─► CLIP text encoder (host) ─► 25 steps × 2 (CFG) = 50 UNet calls ─► VAE decode (host) ─► PNG
                                          │ each call:
                                          │   aiehlc_elfio.py patch  writes this call's inputs into the ELF's .aiehlc_io section
                                          │   apppaltest.py --netload sends the ELF to the board and runs the UNet once
                                          │   reads the result back from the board's output dump and compares cos with onnxruntime on the host
```

Key points:

- **The A78 runs the whole network as C code.** `m.convert` uses TVM to compile the ONNX to C and flattens it into one `graph_driver.c` that calls the layers in order. The A78 on the board is bare-metal; it needs neither Linux nor Python.
- **Offload uses only aiegraph's declaration API.** `emod.match(...)` maps each row of the partition to a TVM call site; `emod.offload_group(...)` declares "this group of calls goes to the AIE, with these steps". aiegraph does the placement, deletion and early-write checks on graph.json and rewrites the driver itself. The user code contains no regexes and does not edit `graph_driver.c`.
- **One launch per layer.** A conv is first turned into a matrix multiply by im2col on the A78, then handed to `gemm_kacc` ([`src/aiednn/gemm/gemm_kacc_kernel.cc`](../../src/aiednn/gemm/gemm_kacc_kernel.cc)): K is accumulated inside the core, and the whole layer is launched once. Attention's QKᵀ, softmax and ×V are fused into one flash-attention kernel ([`src/aiednn/attention/fa_kernel.cc`](../../src/aiednn/attention/fa_kernel.cc)).
- **Keep-layout.** For layers marked `keep_layout` in the partition, the AIE writes its output directly in the layout the next layer wants. The pure-permutation Transpose/Reshape in between is recognized by aiegraph and treated as a "label" only; no data is moved on the A78 (`aiegraph.partition.plan_keep_layout`; the old `keep_layout_codegen.py` has been removed).
- **Runtime inputs.** The inputs live in a dedicated `.aiehlc_io` section of the ELF (`emod.build(runtime_io=...)`). Changing the inputs does not need a rebuild, only a patch of that section.
- **Faster launches.** Warm launch (the kernel is loaded once), control-packet AOT pragmas ([`src/aiednn/gemm/ctrlpkt_aot_pragmas.h`](../../src/aiednn/gemm/ctrlpkt_aot_pragmas.h)), and KCO ([`src/aiednn/gemm/ctrlpkt_kco_pragma.h`](../../src/aiednn/gemm/ctrlpkt_kco_pragma.h); the core configures its own DMA).

---

## 1. Prerequisites

**Hardware**
- A palmyra board with the streaming ("receive while placing") netboot supervisor installed, e.g. palmyra213 (10.23.224.213). The full-network ELF is about 1 GB, which does not fit in the old supervisor's 94 MiB staging buffer.
- This machine can reach the board through `script/test/apppaltest.py --netload`.

**Software**

| What | Path used in this tutorial | Notes |
|---|---|---|
| aiehlc source, `unet_m4` branch | the `aiehlchj` repository | all the code in this tutorial is on this branch |
| LLVM/MLIR build | `/scratch/staff/huaj/llvm-project/build` | to build aiehlc |
| Vitis 2026.x and the aarch64 bare-metal toolchain | found by `script/setup.sh` | same as for any aiehlc use |
| Python environment (numpy, onnx, onnxruntime, Pillow, and aiegraph's dependencies) | `/scratch/staff/huaj/aiehlc/.venv/bin/python3` | used for building and for image generation |
| bk-sdm-tiny ONNX | `/scratch/staff/huaj/modelzoo/artifacts/bk-sdm-tiny/` (`unet_sim.onnx`, `unet.onnx`, `pipeline_ref_256/p00_steps.npz`, `p1/parts/`) | for the build |
| bk-sdm-tiny model zoo | `/scratch/staff/huaj/aiehlc/model_zoo/bk-sdm-tiny/` (text encoder, UNet, VAE, tokenizer) | for image generation |
| `aie_imagegen` (with `palmyra_stage_runner`) | `/scratch/staff/huaj/aiehlc/aiehlc-sdxl-wt/palmyra-runner/script` | aiehlc-sdxl repository, branch `jane/palmyra-stage-runner`, at least 7245e1f (includes the read-back fix) |

The rest of the tutorial uses these variables:

```bash
export AIEHLC=/scratch/staff/huaj/aiehlc/aiehlcopensource/aiehlchj   # repository root
export WORK=$HOME/unet_work                                          # your work directory (inputs, build output, images)
export PY=/scratch/staff/huaj/aiehlc/.venv/bin/python3
mkdir -p $WORK
```

---

## 2. Get the code

Check out the `unet_m4` branch in a worktree:

```bash
cd $AIEHLC
git worktree add .worktrees/unet_m4 unet_m4
export T=$AIEHLC/.worktrees/unet_m4
```

A worktree has no BSP workspace and no private alib; link them from a tree that already has them (skip this step in a full clone):

```bash
bash $T/script/worktree.sh link-deps $T --from $AIEHLC/.worktrees/unet_m2_int6 --private-alib
ls -d $T/thirdparty/arch/*/workspace      # the workspace should be listed
```

Files used in this tutorial:

| File | Purpose |
|---|---|
| [`build_unet.py`](../../example/aiegraph/unet_bk_sdm_tiny/build_unet.py) | build entry point (full listing in section 5): declares the inputs and options, calls aiegraph |
| [`unet_offload.py`](../../example/aiegraph/unet_bk_sdm_tiny/unet_offload.py) | the UNet's offload declaration: conv / linear / attention groups, the keep-layout plan; no C text |
| [`kacc_gemm.py`](../../example/aiegraph/unet_bk_sdm_tiny/kacc_gemm.py), [`aiegraph/kincore.py`](../../src/aiegraph/kincore.py), [`aiegraph/kaccbench.py`](../../src/aiegraph/kaccbench.py), [`aiegraph/flashattn.py`](../../src/aiegraph/flashattn.py), [`aiegraph/attnsite.py`](../../src/aiegraph/attnsite.py) | instantiate the gemm_kacc / flash-attention templates in `src/aiednn` per layer shape (attention + driver glue: generic aiegraph) |
| [`partition_p4a.json`](../../example/aiegraph/unet_bk_sdm_tiny/partition_p4a.json) | which layers go to the AIE |
| [`a78_layer_cflags.txt`](../../example/aiegraph/unet_bk_sdm_tiny/a78_layer_cflags.txt) | compiler flags for the TVM C code on the A78 (NEON) |
| [`export_real_io.py`](../../example/aiegraph/unet_bk_sdm_tiny/export_real_io.py) | exports one real UNet input and the x86 reference output |
| [`run_orange.sh`](../../example/aiegraph/unet_bk_sdm_tiny/run_orange.sh) | image generation: text encoder and VAE on the host, UNet on the board |

Templates used (all under [`src/aiednn/`](../../src/aiednn/), grouped by category):

| Category | Templates | Used for |
|---|---|---|
| gemm | [`gemm_kacc_kernel.cc`](../../src/aiednn/gemm/gemm_kacc_kernel.cc), [`kacc_linear_host.cc`](../../src/aiednn/gemm/kacc_linear_host.cc), [`kacc_unit_head.cc`](../../src/aiednn/gemm/kacc_unit_head.cc) | the AIE kernel and host side of conv and Linear |
| conv2d | [`kacc_conv_host.cc`](../../src/aiednn/conv2d/kacc_conv_host.cc), [`kacc_conv_host_head.cc`](../../src/aiednn/conv2d/kacc_conv_host_head.cc) | conv im2col / scatter |
| attention | [`fa_kernel.cc`](../../src/aiednn/attention/fa_kernel.cc), [`fa_host.cc`](../../src/aiednn/attention/fa_host.cc), [`kacc_bmm_host.cc`](../../src/aiednn/attention/kacc_bmm_host.cc) | flash-attention |
| norm | [`fa_online_softmax.cc`](../../src/aiednn/norm/fa_online_softmax.cc) | the online softmax inside FA |
| elementwise | [`silu_scale_x.cc`](../../src/aiednn/elementwise/silu_scale_x.cc), [`silu_scale_dense.cc`](../../src/aiednn/elementwise/silu_scale_dense.cc) | fused SiLU snippets on the A78 |

For aiegraph itself, see the [aiegraph tutorial](../../doc/tutorial_aiegraph.md).

---

## 3. Build aiehlc

```bash
cmake -S $T -B $T/build \
      -DLLVM_DIR=/scratch/staff/huaj/llvm-project/build/lib/cmake/llvm \
      -DMLIR_DIR=/scratch/staff/huaj/llvm-project/build/lib/cmake/mlir
cmake --build $T/build --target aiehlc -j 32          # about 1 minute
ls -la $T/build/aiehlc
```

You must use the aiehlc built from this branch (it has the two KCO fixes: the `kernel_tm.h` link fix d5ee489, and the per-kernel reset of the KCO configuration plan in multi-kernel programs, 6776083). With another aiehlc, turning on KCO fails at link time or hangs on the board.

---

## 4. Export one real input

The build needs one real input, both as the initial contents of the `.aiehlc_io` section and for the self-check on the board:

```bash
/scratch/staff/huaj/modelzoo/venv/bin/python -I $T/example/aiegraph/unet_bk_sdm_tiny/export_real_io.py \
    --step 0 --branch cond --out $WORK/real_io
```

The last line should be:

```
REAL_IO OK parts=25 data=p00/step0/cond batch_idx=1 t=999 sample_crc32=150c323f unet_in_crc32=a5b911e2 ... sim_vs_full cos=1.0000000
```

`unet_in_crc32=a5b911e2` is the fingerprint of this input; the board prints it later so you can check it.

---

## 5. Build the UNet ELF

### 5.1 All of the user-side source: build_unet.py

This is the full text of `example/aiegraph/unet_bk_sdm_tiny/build_unet.py`. It only declares the model's three inputs, the offload groups (delegated to `unet_offload.py`) and the options. The board main, the inputs header and the PROF/FA driver glue are generated by `aiegraph.boardmain`, and the `.aiehlc_io` section by `emod.build(runtime_io=...)`.

```python
#!/usr/bin/env python3
"""bk-sdm-tiny UNet (unet_sim.onnx, 3 inputs) -> aiegraph -> A78 main.elf, conv / linear / attention on the AIE.

  $PY build_unet.py --onnx unet_sim.onnx --real-io <real_io> --wdtype bf16 --partition partition_p4a.json \
      --aie conv,linear,attn --attn-fa --keep-layout --kincore --launch-opt warm --ctrlpkt aot-kco \
      --arm-layer-cflags-file a78_layer_cflags.txt --runtime-io --dump-out --out <dir>

This file only declares: the model's inputs, the offload groups (unet_offload.py) and the options.
aiegraph generates the board main.c, the inputs header, the PROF / FA driver glue (aiegraph.boardmain)
and the .aiehlc_io runtime-input section (emod.build(runtime_io=...)). The ONNX timestep input is int64;
it is rewritten to int32 + Cast so the board receives exactly the 4 bytes of real_io/timestep_i32.bin.
"""
import argparse
import os
import shutil
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parents[2] / "src"), str(HERE)]   # this checkout's aiegraph (gate: plain $PY)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--onnx", required=True, type=Path)
    ap.add_argument("--real-io", required=True, type=Path, help="sample.bin timestep_i32.bin ehs.bin p1_ref.h")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--aie", choices=("none", "conv", "conv,linear", "conv,linear,attn"), default="none",
                    help="partition layer classes put on the AIE (linear / attn need --kincore)")
    ap.add_argument("--attn-fa", action="store_true", help="each attention group = one flash-attention group")
    ap.add_argument("--keep-layout", action="store_true", help="aiegraph layout folds (needs --attn-fa)")
    ap.add_argument("--wdtype", choices=("fp32", "bf16"), default="fp32", help="weights.bin dtype")
    ap.add_argument("--partition", type=Path, default=None, help="partition.json (layers on the AIE)")
    ap.add_argument("--launch-opt", choices=("cold", "warm"), default="cold", help="warm: kernel loaded once")
    ap.add_argument("--ctrlpkt", choices=("off", "aot", "aot-kco"), default="off", help="control-packet plan")
    ap.add_argument("--kincore", action="store_true", help="one launch per layer, K in the core (gemm_kacc)")
    ap.add_argument("--kincore-layers", default="all", help="--kincore probe subset: partition conv indices")
    ap.add_argument("--arm-layer-cflags-file", type=Path, default=None, help="extra CFLAGS of the TVM layer objects")
    ap.add_argument("--runtime-io", action="store_true", help="inputs in the .aiehlc_io ELF section")
    ap.add_argument("--dump-out", action="store_true", help="--runtime-io: dump the output (AIO_*) on the board")
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args(argv)
    log = lambda msg: print(f"[build_unet] {msg}", flush=True)
    a.ctrlpkt_kco = a.ctrlpkt == "aot-kco"   # aot-kco = the aot plan + KERNELCONFIGOFFLOAD (aiegraph.kincore)
    if a.ctrlpkt_kco:
        a.ctrlpkt = "aot"
    import aiegraph as ag
    from aiegraph import boardmain as bm
    errs = [m for bad, m in (
        (a.aie != "none" and (a.partition is None or a.wdtype != "bf16"), "--aie needs --partition and --wdtype bf16"),
        (a.keep_layout and not a.attn_fa, "--keep-layout needs --attn-fa"),
        (a.dump_out and not a.runtime_io, "--dump-out needs --runtime-io"),
        (a.kincore_layers != "all" and not a.kincore, "--kincore-layers needs --kincore"),
        (a.kincore and a.aie == "none", "--kincore needs --aie conv"),
        (a.ctrlpkt == "aot" and a.aie == "none", "--ctrlpkt aot needs --aie conv"),
        (a.launch_opt == "warm" and a.aie == "none", "--launch-opt warm needs --aie conv")) if bad]
    try:
        layer_cflags = bm.read_cflags_file(a.arm_layer_cflags_file) if a.arm_layer_cflags_file else ""
    except ValueError as e:
        errs.append(f"--arm-layer-cflags-file {e}")
    if errs:
        log("; ".join(errs))
        return 2
    if layer_cflags:
        log(f"arm_layer_cflags ({a.arm_layer_cflags_file}): {layer_cflags}")
    if a.aie != "none":
        bm.profiling_env()
        opts = bm.launch_env(a.launch_opt)
        log(f"launch_opt={a.launch_opt} ({opts}) defs={os.environ['EXTRA_DEFS']} "
            f"aiehlc.sh={os.environ.get('AIEHLC_SH_ARGS', '')}")
    out, real_io = a.out.resolve(), a.real_io.resolve()
    out.mkdir(parents=True, exist_ok=True)
    root = Path(ag.__file__).resolve().parents[2]
    os.environ.setdefault("AIEHLC_ROOT", str(root))
    os.environ["AIEGRAPH_PARAMS_INPLACE"] = "1"         # 1.29 GB weights read in place from the linked blob
    os.environ["AIEGRAPH_WDTYPE"] = a.wdtype
    os.environ.setdefault("HOSTCOMPILE_HOST_OPT", "-O2")  # aiek host.cc (im2col / scatter glue inside graph_run)
    log(f"aiegraph from {ag.__file__} (repo {root}); HOSTCOMPILE_HOST_OPT={os.environ['HOSTCOMPILE_HOST_OPT']}")

    # the model's inputs, in graph order: (graph name, C name, real_io file)
    ins = [("sample", "sample", "sample.bin", "<f4"), ("timestep", "timestep", "timestep_i32.bin", "<i4"),
           ("encoder_hidden_states", "ehs", "ehs.bin", "<f4")]
    arrs = [(g, c, np.fromfile(real_io / f, dtype=dt)) for g, c, f, dt in ins]
    assert [x.size for _, _, x in arrs] == [4 * 32 * 32, 1, 77 * 768], [x.size for _, _, x in arrs]
    hdr = out / "unet_inputs.h"
    crc = bm.inputs_header(hdr, "unet", arrs, dict(AIE_MODE=a.aie, WDTYPE=a.wdtype), origin=real_io)
    want = (real_io / "unet_in.crc32").read_text().strip()
    if crc != want:
        log(f"inputs crc32 {crc} != {real_io / 'unet_in.crc32'} {want}")
        return 1
    log(f"inputs: {', '.join(f'{g} {x.size} {x.dtype}' for g, _, x in arrs)}; crc32={crc} == unet_in.crc32")

    onnx_i32 = bm.input_int32(a.onnx.resolve(), out / "unet_sim_t32.onnx", "timestep", log=log)
    m = ag.load(str(onnx_i32))
    log(f"loaded {m.name}: inputs {m.input_shapes} -> {m.output_shape}, {sum(m.ops.values())} nodes")
    shutil.rmtree(out / "ag_work", ignore_errors=True)   # a fresh conversion every build
    emod = m.convert(target="a78", quantize="fp32", image=str(real_io / "sample.bin"),
                     workdir=out / "ag_work", verbose=a.verbose)
    log(f"converted -> {emod.workdir}")
    bm.cross_toolchain(root, log=log)
    n_patched = None
    if a.aie != "none":
        import unet_offload
        defs, n_patched = unet_offload.offload(a, emod, out)
        bm.append_defines(hdr, "unet", defs)
    main_c = bm.main_c("unet", [Path(f).stem for _, _, f, _ in ins], aie=a.aie != "none",
                       runtime_io=a.runtime_io, note=(1, "timestep"))
    rio = [(Path(f).stem, real_io / f) for _, _, f, _ in ins] if a.runtime_io else None
    rep = emod.build(target="a78", verbose=a.verbose, main_c=main_c,
                     extra_sources=[bm.cmp_source(), real_io / "p1_ref.h", hdr],
                     extra_cflags=f"{bm.CMP_CFLAGS} {bm.HEAP_LDFLAG}", layer_cflags=layer_cflags,
                     runtime_io=rio, dump_out=a.dump_out, launch=a.launch_opt)
    if not rep.ok:
        log(f"build FAILED: {rep.reason}")
        return 1
    for ln in rep.io_info.splitlines():
        log(ln)
    if n_patched is not None:
        unet_offload.check_driver((Path(rep.elf).parent / "graph_driver.c").read_text(), n_patched)
    elf, wb = out / "main.elf", Path(rep.elf).parent / "weights.bin"
    shutil.copy2(rep.elf, elf)
    if wb.is_file():   # the blob netload streams to the board, beside the ELF
        shutil.copy2(wb, out / "weights.bin")
    log(f"built {rep.elf} -> {elf} ({elf.stat().st_size:,} B; weights.bin "
        f"{wb.stat().st_size if wb.is_file() else -1:,} B)")
    return 0


if __name__ == "__main__":
    try:
        rc = main()
    except SystemExit as e:
        rc = e.code if isinstance(e.code, int) else (print(e.code, flush=True) or 1)
    except Exception:
        import traceback
        traceback.print_exc()
        rc = 1
    print(f"GATE_RC={rc}", flush=True)
    sys.exit(rc)
```

These are the only things it asks of aiegraph:

| Step | aiegraph call | What it does |
|---|---|---|
| Load, convert | `ag.load(...)`, `m.convert(target="a78", ...)` | ONNX → TVM C (one function per layer) + graph.json |
| Inputs | `bm.inputs_header(...)`, `bm.input_int32(...)` | inputs header + crc; the ONNX int64 timestep becomes int32 + Cast |
| Match, declare groups | inside `unet_offload.offload(a, emod, out)`: `emod.match("conv"/"linear", partition=...)`, `emod.offload_group(g, kcc, steps=[ag.step(...)])` | conv / linear: one group per call; flash-attention is a multi-call group (pack Q/K/V + run O; the softmax and other calls are deleted by aiegraph) |
| Weights | `weights=g.w.as_bf16(layout=...)` | weights.bin holds the bf16 layout the kernel wants, read in place |
| Driver template | `emod.driver_template(**bm.driver_glue(...))` | FA declarations, PROF timing macros; aiegraph places them at a fixed anchor (the anchor must be unique; applying twice is an error) |
| Check, build | `emod.offload(..., check="group")`, `emod.build(target="a78", main_c=bm.main_c(...), runtime_io=...)` | group-level x86 check (TVM C chain vs steps, cos ≥ 0.999); one library, one `routing()` |

### 5.2 Run it

```bash
cd $T/example/aiegraph/unet_bk_sdm_tiny
env -u AIEHLC_ROOT AIEHLC_BIN=$T/build/aiehlc $PY build_unet.py \
    --onnx /scratch/staff/huaj/modelzoo/artifacts/bk-sdm-tiny/unet_sim.onnx \
    --real-io $WORK/real_io \
    --wdtype bf16 \
    --partition partition_p4a.json \
    --aie conv,linear,attn --attn-fa --keep-layout --kincore \
    --launch-opt warm --ctrlpkt aot-kco \
    --arm-layer-cflags-file a78_layer_cflags.txt \
    --runtime-io --dump-out \
    --out $WORK/build 2>&1 | tee $WORK/build.log
```

What each switch does:

| Switch | Effect |
|---|---|
| `--wdtype bf16` | store the weights as bf16 (0.65 GB); bf16 multiplies and fp32 accumulation on the AIE |
| `--partition partition_p4a.json` | which layers go to the AIE |
| `--aie conv,linear,attn` | put the conv, Linear and attention classes on the AIE |
| `--attn-fa` | fuse attention's QKᵀ, softmax and ×V into one flash-attention group |
| `--keep-layout` | the AIE writes its output in the layout the next layer wants, removing the layout transforms on the A78 |
| `--kincore` | one launch per layer, K accumulated inside the core |
| `--launch-opt warm` | load the kernel once, then launch it repeatedly |
| `--ctrlpkt aot-kco` | control-packet AOT pragmas, plus KCO |
| `--arm-layer-cflags-file` | NEON compiler flags for the TVM C code on the A78 |
| `--runtime-io --dump-out` | put the inputs in the `.aiehlc_io` section and dump the output to the console; needed for image generation |

It takes about 25 minutes (measured: 1645 s). The log should contain:

```
[build_unet] loaded unet_sim_t32: inputs {'sample': [1, 4, 32, 32], 'timestep': [1], 'encoder_hidden_states': [1, 77, 768]} -> [1, 4, 32, 32], 828 nodes
[build_unet] KINCORE layers=156 launches_planned=258 kernel=bf16_gemm_kacc; shapes=38 (old 64x64x64 launches 216200)
[build_unet] ctrlpkt=aot pragmas=6
[build_unet] graph_driver.c: 156 partition call sites -> aiednn_aie_conv (48) + aiednn_aie_linear (72) + aiednn_aie_attn (36) (aiegraph group steps + driver template)
[build_unet] AIO section off=0x403b7100 addr=0x403a7100 bytes=253248 n_in=3 crc32=a5b911e2 dump_out=1
[build_unet] built .../main.elf -> $WORK/build/main.elf (1,080,180,080 B; weights.bin 646,757,376 B)
GATE_RC=0
```

To look at what was generated:
- the graph driver after aiegraph's rewrite: `$WORK/build/ag_work/arm_build/graph_driver.c` (search for `aiednn_aie_conv(`);
- the placement of every layer: `$WORK/build/ag_work/layers/partition.json`;
- all AIE kernels and host functions: `$WORK/build/aiek_src/`.

---

## 6. Run the UNet once on the board (smoke test)

First run once with the input built into the ELF, to confirm the ELF is correct on the board:

```bash
cd $T
$PY script/test/apppaltest.py -board palmyra -host 10.23.224.213 -y --netload \
    --netload-wait 1800 --netload-reply-timeout 600 --netload-max-run-ms 300000 \
    $WORK/build/main.elf 2>&1 | tee $WORK/smoke.log | grep -v DEBUG | \
    grep -E "^Host: |\[netboot\] run|in_crc32_board|^unet: graph_run|^LAUNCH warm|^CMP unet"
```

You should see (measured 2026-11-18 on unet_m4):

```
unet: graph_run 1295.310 ms, output elems=4096 (ref 4096)
LAUNCH warm ms_per_launch=2.945 launches=258 aie_wall_ms=759.738 ... ctrlpkt=aot aot_fallbacks=0
CMP unet board=palmyra213 data=p00/step0/cond in_crc32=a5b911e2 cos=1.000000 maxabs=0.00324592 dtype=bf16 aie_launches=258 coreen_calls=258 run_rc=0 aie_flop_frac=1.000 PASS
```

How to read these lines:
- `in_crc32=a5b911e2`: the board read exactly the input you gave it;
- `aie_launches=258 coreen_calls=258`, `aot_fallbacks=0`: the AIE really was launched 258 times, with no silent fallback to the A78;
- `cos=1.000000 … PASS`: matches the onnxruntime result on x86;
- `graph_run 1295.310 ms`: the time of one UNet call (760 ms of it on the AIE).

**Note:** `apppaltest.py` returns 0 even when the board prints FAIL, so always check the `CMP … PASS` line, not just the return code.

---

## 7. Generate the orange

```bash
cd $T/example/aiegraph/unet_bk_sdm_tiny
./run_orange.sh $WORK/build/main.elf $WORK/img 2 25      # seed 2, 25 steps: all 50 UNet calls on the board
```

`run_orange.sh` does three things:
1. It creates a private model zoo in `$WORK/img/zoo/bk-sdm-tiny/` and writes an `aie/aie_manifest.json` that hands the UNet step to the board runner (`aie_imagegen.palmyra_stage_runner`, which patches the `.aiehlc_io` section with `src/tool/elfio/aiehlc_elfio.py`; `--map timestep=timestep_i32:int32 --map encoder_hidden_states=ehs` maps the input names to the ELF's input slots).
2. It runs `aie_imagegen --backend aie`: the text encoder and the VAE run on the host; every UNet call is netloaded to the board, run, and its result read back. There is no host fallback: if any board call fails, the image fails.
3. It summarizes: one `STEP …` line per call in `$WORK/img/steps_s2.log` (the input crc the board read, `graph_run_ms`, and the cos against onnxruntime).

Parameters: prompt "a photo of an orange, citrus fruit, high quality", negative prompt "blurry, deformed, low quality", 256×256, guidance 7.5. Override them with the environment variables `PROMPT`, `NEG`, `GUIDANCE` and `BOARD`.

**Compare with the host version (optional).** First generate a reference image on the host with the same parameters, then let `run_orange.sh` also compute PSNR and cos:

```bash
cd $WORK && PYTHONPATH=/scratch/staff/huaj/aiehlc/aiehlc-sdxl-wt/palmyra-runner/script $PY -m aie_imagegen \
    -p "a photo of an orange, citrus fruit, high quality" --negative-prompt "blurry, deformed, low quality" \
    -o $WORK/host_seed2.png --backend host --zoo /scratch/staff/huaj/aiehlc/model_zoo \
    --model bk-sdm-tiny --size 256 --steps 25 --seed 2 --guidance 7.5 --decoder vae
cd $T/example/aiegraph/unet_bk_sdm_tiny
REF=$WORK/host_seed2.png ./run_orange.sh $WORK/build/main.elf $WORK/img 2 25
```

---

## Measured results

Measured 2026-11-18 on palmyra213 (unet_m4, pure aiegraph flow), seed 2, 25 steps:

```
[run_orange] UNet calls on the board: 50, cos_OK: 50, mean graph_run_ms: 1307.9
[run_orange] vs seed2.png: PSNR 20.11 dB, cos 0.97028
```

| Metric | Value |
|---|---|
| Smoke (one UNet call) | `graph_run` 1295.3 ms, launches 258, `aot_fallbacks=0`, cos 1.000000 PASS |
| UNet calls | 50 (25 steps × CFG 2), all on the board, no fallback, 50/50 cos_OK |
| Mean `graph_run` | 1307.9 ms |
| Whole image vs host (fp32) | PSNR 20.11 dB, cos 0.97028 |
| Wall clock for the whole image | 2542 s, almost all of it netload of the 1 GB ELF, not compute |

Left: generated on the board. Right: generated on the host by onnxruntime (fp32) with the same parameters:

![Board vs host](tiny-sdm-unet-orange-vs-host.png)

The two images have the same composition and colors. The whole-image PSNR is only 20 dB because the weights on the AIE are bf16: each step's small error is accumulated and amplified over the 25 denoising steps, so details drift. The per-call cos is the criterion for "the board computed it correctly"; the whole-image PSNR only tells you whether the run went off track.

**Why the wall clock is 40 minutes, not 50 × 1.3 s ≈ 65 s.** Today every UNet call netloads the whole ELF (including 0.65 GB of weights) to the board again, just to change 250 KB of input. To get down to about a minute, the ELF has to stay resident on the board and each step should transfer only its inputs and outputs.

---

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `apppaltest` does not move and prints `[devlock] waiting` | someone else is using the board. It queues automatically and continues once it gets the lock within `--netload-wait` |
| `elf_size … exceeds staging 98566144` | the board still has the old netboot supervisor; switch to the streaming version |
| the KCO kernel fails to link during the build (`__aiehlc_tm_anchor is undefined`) | the aiehlc is not the one built from this branch. Point `AIEHLC_BIN` at `$T/build/aiehlc` |
| with KCO on, the board hangs at the 4th launch (`wait_io TIMEOUT`) | same as above: an old aiehlc without the 6776083 fix |
| the build fails with errors about `thirdparty/alib`, or two builds break each other | the two builds share one alib. Use `link-deps … --private-alib` in the worktree, or don't build concurrently |
| make reports odd archiver errors | `AR` is exported in the environment, and make uses it as the archiver. `unset AR` before building |
| `[DEBUG]` prints are mixed into the result lines and the regex does not match | `apppaltest`'s debug output; filter it with `grep -v DEBUG` first |
| `STEP k=0 rc=4`, `readout: 16384 B of output (need 236544 B)` | the runner does not know the output shape. Add `--out-shape 1,4,32,32` (`run_orange.sh` already does) |
| `STEP … cos_OK=0` | look at that call's `calls_s2/call_<k>/console.log`: does the input crc MATCH, and is there a `CMP … PASS` |

---

## Next steps: where to change things

| To do this | Change |
|---|---|
| put a different set of layers on the AIE | `partition_p4a.json`, or regenerate it with [`src/aiegraph/autopart.py`](../../src/aiegraph/autopart.py) |
| change how the gemm_kacc kernel is written | [`src/aiednn/gemm/gemm_kacc_kernel.cc`](../../src/aiednn/gemm/gemm_kacc_kernel.cc) |
| change flash-attention | [`src/aiednn/attention/fa_kernel.cc`](../../src/aiednn/attention/fa_kernel.cc) and [`fa_host.cc`](../../src/aiednn/attention/fa_host.cc) |
| change the board main / PROF glue | [`src/aiegraph/boardmain.py`](../../src/aiegraph/boardmain.py) |
| test one kernel on its own | [`kacc_bench.sh`](../../example/aiegraph/unet_bk_sdm_tiny/kacc_bench.sh) `--gemm` / `--bmm` |
| see where the time goes | the `LAUNCH split` and `PROF` lines in the board log |
| move a different CNN model | see [ResNet18](resnet18.md) and [YOLOv26](yolov26.md): the same aiegraph, with conv/gemm all offloaded at once by `aiegraph.cnn` / `aiegraph.cnn_chunk` |

Not done yet: GroupNorm and LayerNorm still run on the A78, conv im2col still runs on the A78, all data between layers goes through DDR, every launch is synchronous, and only 16 of the 144 tiles are used.
