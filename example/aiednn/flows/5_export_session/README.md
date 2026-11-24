# Flow 5: `torch.export` → session API (you write `main`)

```
torch.export  →  dnn_export.to_fx / dnn_torch  →  emit_model  →  <name>_model.h/.cc
                                                              →  your host.cc  →  dnn_build → ELF
```

Unlike flow 2 `--export` / flow 4 `--app` (which emit a full benchmark `main`), this path
writes a **session library**:

```c
#include "tiny_model.h"
tiny_t *m = tiny_create();
tiny_run(m, input, output);          // layouts: tiny_input_desc(m) / tiny_output_desc(m)
tiny_destroy(m);
```

[`host.cc`](host.cc) is that hand-written host (placeholders filled by `run.py`).

```bash
python example/aiednn/flows/5_export_session/run.py [--model tiny]
python example/aiednn/flows/5_export_session/run.py --target vek280 --board galerina13 \
    --vitis-settings <.../Vitis/settings64.sh>
```

```
exported: .../tiny.pt2
session:  .../tiny_model.h
=== flow 5 export session (tiny) ===
input ndims=4 dt=4  output ndims=2 dt=4
top-1 8 (reference 8)  rel err ...  logits[0]=...
PASS: flow 5 export session
PASS: flow 5 export session [tiny] (int8) -> x86 host-owned main
```

The host checks its output against the PyTorch reference that `run.py` writes (`<model>_golden.bin`):
relative L2 error under the mode's tolerance and the same top-1, else `FAIL`. Artifacts land in
`worklocal/aiednn/flows/flow5_<model>_<mode>/`. Needs a current `aiehlc` (`AIEHLC_BIN=<build>/aiehlc` if
`release/prebuild` is too old for `gemm_uk`).

Also available as `compile_exported(..., emit="model")` and flow 2 `--export --emit model`
(those build a thin smoke host instead of this hand-written one).
