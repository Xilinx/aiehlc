# Flow 2: `torch.compile(model, backend="aiednn")`

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

**On a board:** `example/aiednn/models/torch_compile_board.py <model> pt2e` emits the same partitions as a C
app (`aiegraph.dnn_graph_app`), builds it for VEK280 / VEK385 and checks it there against the workstation run
of the same graph.
