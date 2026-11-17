# Flow 2: PyTorch in: `torch.compile(model, backend="aiednn")` and `torch.export`

```python
import aiegraph.backends.aiednn as flags      # registers the backend; flags like torch.backends.cudnn
y = torch.compile(model, backend="aiednn")(x)
```

Dynamo passes the fx graph to `aiegraph.dnn_torch_backend`. `dnn_torch_graph.FxGraph` maps each fx node
to one graph op, freezing parameters and folding Conv+BN. The partition graph API then forms partitions,
and each supported partition replaces its fx nodes with a module that executes the compiled partition.
Everything else stays torch code.

```bash
python example/aiednn/flows/2_torch_compile/run.py [--model resnet18]
```

```
4 partitions: 4 supported, 0 unsupported; 7 / 7 fx nodes inside supported partitions
  + conv+relu -> gemm+conv+bf16x3:aie:bf16_f32
  + conv+add+relu -> gemm+conv+bf16x3:aie:bf16_f32
  + reduction -> fvec:aie:fvec_f32
  + matmul -> bf16x3:aie:bf16_f32
PASS: flow 2 torch.compile, max rel diff vs eager 2.24e-07
```

**Flags** (`aiegraph.backends.aiednn`):
- `benchmark`: time every implementation once and keep the fastest.
- `graph`: use the partition graph API.
- `mode` / `mlir`: the older region path.

For int8, quantize with PT2E first; the dequantize / quantize ops become int8 primitives (fusion level 1
is bit-identical to the unfused ops, level 2 within tolerance; see `aiednn_framework_api.md` §1 / §7).

## `torch.export` -> a board program

```python
from aiegraph import dnn_export
ep = torch.export.export(dnn_export.prepare_for_export(model), (x,))   # an ExportedProgram (save with torch.export.save)
rep, g, _ = dnn_export.compile_exported(ep, x, mode="int8", target="vek385",
                                        vitis_settings=".../Vitis/settings64.sh")  # the board ELF: rep.elf
dnn_export.run(rep, board="<host>")              # separately, on any VEK385 (x86: run(rep) runs it here)
```

```bash
python example/aiednn/flows/2_torch_compile/run.py --export [--model resnet18]          # x86 C app, run here
python example/aiednn/flows/2_torch_compile/run.py --export --model gpt2_small --target vek385 \
    --board portobello13 --vitis-settings <.../Vitis/settings64.sh>                     # board ELF, run there
python example/aiednn/flows/2_torch_compile/run.py --pt2 model.pt2 --model resnet18      # a saved program
```

The exported program goes through the **offline model compiler** (flow 4), not the partition graph API:
`aiegraph.dnn_export.to_fx` turns its ATen graph back into the torch-level graph flow 4 lowers -- ops with
parameters become the modules they came from, rebuilt from the program's tensors (`aten.convolution` ->
`nn.Conv2d`, `addmm` / `mm` of a transposed constant -> `nn.Linear`, the norm ops -> `nn.BatchNorm2d` /
`nn.LayerNorm`), every other op the torch function of the same name, input-independent nodes are evaluated
(yolov8n's anchor grid), and two export idioms are folded back: `view -> Linear -> view` around every Linear on
a 3-D input, and GQA's `unsqueeze -> expand -> view` (-> `repeat_interleave`). Then dnn_torch lowers it (int8
calibrated on `x`), the MLIR pipeline fuses it, `emit_app` writes the C app and `dnn_build` compiles it with
aiehlc into an ELF carrying the AIE megakernel. `prepare_for_export` only swaps `nn.MultiheadAttention` for the
same math in fusable modules (vit_b_16: 116 primitives instead of 284); a program exported without it compiles
too. Workstation emulation, int8, MLIR, 2026-11-16 -- the same primitives as `run_models.py --mlir` (direct):

| model | primitives (on AIE) | int8 GEMMs exact | rel err vs PyTorch | top-1 |
|---|---|---|---|---|
| resnet18 | 25 (25) | 21/21 | 2.0e-2 | 258 |
| mobilenet_v2 | 56 (56) | 53/53 | 3.2e-2 | 258 |
| gpt2_small | 109 (109) | 72/72 | 1.7e-2 | 1004 |
| tiny_llama | 57 (49) | 28/28 | 8.6e-3 | 4492 |
| vit_b_16 | 116 (113) | 74/74 | 0.12 | 258 |
| bert_base | 111 (110) | 73/73 | 0.11 | 27 |
| yolov8n | 150 (104) | 64/64 | 2.1e-2 | 5 (bus) |

**Graph-API board apps (PT2E):** `example/aiednn/models/torch_compile_board.py <model> pt2e` emits the same partitions as a C
app (`aiegraph.dnn_graph_app`), builds it for VEK280 / VEK385 and checks it there against the workstation run
of the same graph.
