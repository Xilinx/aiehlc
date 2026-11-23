# Flow 4: the offline model compiler

This is the whole-model path behind `example/aiednn/models/run_models.py`:

1. `aiegraph.dnn_torch` lowers the PyTorch model to an `AGraph` (f32, or int8 calibrated on the input).
2. The aiednn MLIR dialect and passes (`aiegraph.dnn_mlir`, `aiednn-opt`) optimize it: canonicalize,
   fuse-epilogue, quantize-residual, fold-requant, fold-head-permute, plan-memory.
3. The result is a sequence of fused primitives that calls only `include/aiednn.h`. It runs through ctypes
   on the workstation, or becomes a C app that `ag.dnn_build` turns into a board ELF.

```bash
python example/aiednn/flows/4_model_compiler/run.py [--model resnet18]                    # ctypes, workstation emulation
python example/aiednn/flows/4_model_compiler/run.py --app              # the C app, built and run on x86
python example/aiednn/flows/4_model_compiler/run.py --app --target vek280 --board <host> \
    --vitis-settings <.../Vitis/settings64.sh>                           # the C app on the board
```

```
6 primitives, 6 on the AIE; integer GEMMs bit-exact: 3/3
    2  steps:aie:lut8
    2  im2col+blocked:aie:gemm_s8s32
    1  fvec:aie:fvec_f32
    1  blocked:aie:gemm_s8s32
PASS: flow 4 model compiler (int8, mlir), rel err vs PyTorch 7.16e-08, top-1 8 (reference 8)
```

The C app re-runs every integer conv / matmul on `ref:` on the device and compares bit for bit
(`gemm_checked` / `gemm_bad` on its MODEL line). For real models with board numbers, use `run_models.py` with
`--app vek280|vek385` (README in `../../models/`).
