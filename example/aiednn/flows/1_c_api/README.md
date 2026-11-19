# Flow 1: the C primitive API

`matmul_relu_app.cc` is a complete aiednn application: an s8 matmul (256×256×256) and a relu, each a
primitive. The app describes tensors (`aiednn_memory_desc_init`), creates primitives
(`aiednn_matmul_primitive_create`, `aiednn_eltwise_forward_create`) and executes them. It never names a
kernel. Dispatch picks the implementation: the AIE GEMM for the matmul, the AIE table mode for the relu.
The app then repeats both calls with `aiednn_set_impl_filter("ref:")` and compares.

```bash
# workstation emulation
PYTHONPATH=src python -m aiegraph dnn-build example/aiednn/flows/1_c_api/matmul_relu_app.cc --target x86 --run
# a board: builds the ELF with aiehlc, programs the board, runs it
PYTHONPATH=src python -m aiegraph dnn-build example/aiednn/flows/1_c_api/matmul_relu_app.cc \
    --target vek280 --run --board <host> --vitis-settings <.../Vitis/settings64.sh>
```

```
matmul -> blocked:aie:gemm_s8s32
relu   -> lut:aie:lut8
PASS: flow 1 C API, matmul on blocked:aie:gemm_s8s32, relu on lut:aie:lut8, bit-exact vs ref:
```

`dnn-build` scans the source for `*_create` calls to choose which kernels to link, so a new op is a new
create call. Without aiehlc in the loop, link against the prebuilt library instead (`../../prebuilt/`).
API reference: `include/aiednn.h`; design: `doc/design/aiednn.md`.
