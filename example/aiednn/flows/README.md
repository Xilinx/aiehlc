# aiednn: the architecture and the four ways in

One small, runnable example for each way a program or model reaches the AI Engine through aiednn. All
four end in the same library (`include/aiednn.h`), the same dispatch and the same AIE megakernel. The
deeper references are:

- [`doc/design/aiednn_architecture.md`](../../../doc/design/aiednn_architecture.md): the one-page overview with diagrams and a file map.
- [`doc/design/aiednn.md`](../../../doc/design/aiednn.md): the library and the model compiler.
- [`doc/design/aiednn_framework_api.md`](../../../doc/design/aiednn_framework_api.md): the graph API, the framework hooks, op parity and board results.

```
            1 C app            2 torch.compile           3 ONNX                 4 model compiler
         (aiednn.h calls)    (backend="aiednn")    (OnnxGraph / ORT EP)     (run_models / dnn_mlir)
               │                    │ fx graph              │ ONNX graph             │ fx → AGraph → MLIR passes
               │                    ▼                       ▼                        │ (fusion, requant folds,
               │          ┌─ PARTITION GRAPH API (aiednn_graph.h) ─┐                │  memory plan)
               │          │ add ops → get_partitions → compile →   │                │
               │          │ execute; supported = the fused         │                │
               │          │ primitive creates; the rest stays in   │                │
               │          │ the framework                          │                │
               │          └──────────────────┬─────────────────────┘                │
               ▼                             ▼                                      ▼
 ╔═════════════════════ LIBRARY: include/aiednn.h, src/aiednn/ ══════════════════════════════╗
 ║ describe tensors → *_create → DISPATCH over the impl list (generated from AIEK_SPECs):      ║
 ║ AIE first, most specific acceptor wins, ref: (CPU) last → execute (host pack, launch, wait) ║
 ╚══════════════════════════════════════════════╤═════════════════════════════════════════════╝
                                                ▼
       AIE megakernel ELF (src/aietensorop/gemm/gemm_uk.cc + aie_modes_*.h): GEMM, CONV, DW, POOL,
       LUT, STEPS, FVEC, BF16, SDPA, … modes, selected per launch  →  VEK280 (A72) / VEK385 (A78)
```

| # | Way in | Who uses it | Entry point | This directory | What it prints |
|---|---|---|---|---|---|
| 1 | **C primitive API**: describe tensors, create primitives, execute | apps, runtimes, other compilers | `include/aiednn.h` | [`1_c_api/`](1_c_api/) | implementation per primitive, PASS vs `ref:` |
| 2 | **`torch.compile(model, backend="aiednn")`**, and **`torch.export` -> a board program** (`--export`) | PyTorch users | `aiegraph.dnn_torch_backend` → graph API; `aiegraph.dnn_export` → the model compiler (flow 4) | [`2_torch_compile/`](2_torch_compile/) | partitions + implementations, PASS vs eager; with `--export`: the C app / board ELF, PASS vs PyTorch |
| 3 | **ONNX**: `OnnxGraph`, or the ONNX Runtime plugin EP | ONNX / ORT users | `aiegraph.dnn_onnx_graph`, `aiegraph.dnn_ort_ep` → graph API | [`3_onnx/`](3_onnx/) | partitions, PASS vs onnxruntime CPU |
| 4 | **Offline model compiler**: a whole model → fused primitive sequence → board app | per-model deployment, benchmarks | `aiegraph.dnn_torch` + `dnn_mlir` | [`4_model_compiler/`](4_model_compiler/) | primitives on the AIE, GEMMs checked exact, PASS vs PyTorch |

Flows 2 and 3 go through the **partition graph API**. The library decides what it can take: an anchor op
(conv, matmul) absorbs its bias, activations, residual add and requantization, and a partition is
supported when its fused primitive creates. Flow 4 does the whole-model work offline (MLIR passes) and emits
plain `aiednn.h` calls. Flow 1 is those calls written by hand.

## Which model

- **Flow 1** has no model: two primitives on synthetic int8 data in the C file.
- **Flows 2–4** default to `--model tiny`: the small net in [`model.py`](model.py) (conv-bn-relu, conv with a
  residual add, global pool, fc; random weights and input), so the whole set runs in seconds and every
  partition fits on one screen.
- **Real models:** `--model <name>` loads them from [`../models/zoo.py`](../models/zoo.py): torchvision
  models with their ImageNet weights on a real photo (`resnet18`, `mobilenet_v2`, `resnet50`, `vit_b_16`, …),
  and `bert_base`, `gpt2_small`, `tiny_llama`, `yolov8n`. Flow 3 also takes an existing ONNX file
  (`--onnx`), e.g. onnxruntime's int8 QDQ ResNet-18:

```bash
python example/aiednn/flows/2_torch_compile/run.py --model resnet18
python example/aiednn/flows/2_torch_compile/run.py --export --model resnet18 [--target vek385 --board <host>]   # torch.export
python example/aiednn/flows/3_onnx/run.py --onnx worklocal/tvmrelay_deploy/resnet18-v1-7.int8.qdq.onnx
python example/aiednn/flows/4_model_compiler/run.py --model resnet18          # int8, the MLIR pipeline
```

ResNet-18 (ImageNet weights, the Samoyed photo, class 258), workstation emulation, 2026-11-16:

| Flow | Partitions / primitives | vs reference | top-1 |
|---|---|---|---|
| 2 `--model resnet18` (f32) | 23 partitions, 49 / 49 fx nodes supported | 2.2e-6 vs eager | 258 |
| 3 `--model resnet18` (exported f32 ONNX) | 23 partitions, 49 / 49 nodes | 2.5e-6 vs ORT CPU, OnnxGraph and the EP | 258 |
| 3 `--onnx resnet18-v1-7.int8.qdq.onnx` | 26 partitions, 140 / 140 nodes | OnnxGraph 0.0, EP 1.1e-2 vs ORT CPU | 258 |
| 4 `--model resnet18` (int8, MLIR) | 25 primitives, all on the AIE, 21 / 21 int8 GEMMs bit-exact | 2.0e-2 vs PyTorch f32 | 258 |

## Run all four (workstation emulation)

The AIE kernels' C references stand in for the array, so `aie:` in an implementation name means the
AIE path ran in emulation. From the repo root, with a Python that has torch, onnx and onnxruntime >= 1.23
(e.g. `/scratch/staff/$USER/aiednn_venv/bin/python`):

```bash
bash example/aiednn/flows/run_all.sh
```

Each flow ends in `PASS: flow N ...`. Expected, as run on 2026-11-16:

```
PASS: flow 1 C API, matmul on blocked:aie:gemm_s8s32, relu on lut:aie:lut8, bit-exact vs ref:
PASS: flow 2 torch.compile, max rel diff vs eager 2.24e-07
PASS: flow 3 ONNX
PASS: flow 4 model compiler (int8, mlir), rel err vs PyTorch 7.16e-08, top-1 8 (reference 8)
```

## On a board

Each flow's README has the board command. Flows 1 and 4 build a board ELF with
`python -m aiegraph dnn-build` / `ag.dnn_build` (`--target vek280|vek385 --board <host>`). Flows 2 and 3 run
their partitions in-process on the workstation. Their board form is
[`../models/torch_compile_board.py`](../models/torch_compile_board.py): the same graph-API partitions, emitted
as a C app and checked on the board against the workstation run (`aiegraph.dnn_graph_app`).

## More examples

| | |
|---|---|
| [`../matmul_app.cc`](../matmul_app.cc), [`../ops_app.cc`](../ops_app.cc), [`../lut_app.cc`](../lut_app.cc), [`../fvec_app.cc`](../fvec_app.cc), [`../wq_app.cc`](../wq_app.cc) | more C API apps, one op family each |
| [`../bench_app.cc`](../bench_app.cc), [`../parity_bench.cc`](../parity_bench.cc) | per-op AIE vs `ref:` scoreboards |
| [`../hostflow/`](../hostflow/) | your own C program around graph partitions, with one op kept on the host |
| [`../prebuilt/`](../prebuilt/) | an app against the prebuilt static library (`python -m aiegraph dnn-lib`, `link.mk`) |
| [`../models/`](../models/) | `run_models.py`: the zoo (ResNet, MobileNet, ViT, BERT, GPT-2, TinyLlama, YOLOv8n, …) through flow 4 |
| [`../resnet18/`](../resnet18/) | TVM's int8 ResNet-18 as aiednn primitives, bit-identical to TVM |
