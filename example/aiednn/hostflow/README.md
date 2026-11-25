# A host program around a small model

`host_app.cc` is the whole program. It owns the inputs, the outputs, and one step of the model. The library runs the other two.

The model is three ops on a 2×8 input and a 4×8 weight matrix:

| step | who runs it | what it does |
|---|---|---|
| matmul | library | `y = x @ Wᵀ` |
| scale | this file | `u = 0.5 * y` |
| relu | library | `z = relu(u)` |

`main` builds that graph once. The library splits it into partitions and compiles the ones it can run. The scale is a `wildcard`, so it stays in this file. Then, for each image, `main` fills `x`, walks the partitions in order, and reads `z`. The weights are marked constant and are not rewritten.

The same calls are what `run_models.py` and `torch_compile_board.py` generate. Those apps fill in `main` for a benchmark. This one is the `main` you write.

## Build and run

From the repo root:

```bash
PYTHONPATH=src python -m aiegraph dnn-build example/aiednn/hostflow/host_app.cc --target x86 --run
```

A passing run prints one line per partition, one `image N max_diff` line per image, and `PASS: hostflow`. The numbers are checked against the float matmul in `expect()`.

On a board, the same file:

```bash
PYTHONPATH=src python -m aiegraph dnn-build example/aiednn/hostflow/host_app.cc \
    --target vek280 --run --board <host> --vitis-settings <.../Vitis/settings64.sh>
```

A passing run looks like this:

```
part 0 library  ref:matmul
part 1 host     wildcard unsupported: wildcard op
part 2 library  ref:eltwise
image 0 max_diff 0
image 1 max_diff 0
PASS: hostflow, 2 images
```

This matmul is f32 and 2×8×4, and the linked kernel accepts int8 256×256, so dispatch keeps it on `ref:`. An int8 matmul that kernel accepts prints an `aie:` implementation on the same line. The host loop does not change.

## What to change

The images and `SCALE` are the host logic. `g_x_img` is filled in `main`; point `g_x` at whatever buffer the next call should read.

The model is the three `aiednn_graph_add_op` calls in `build()`. Another library op is another call, another id, and another buffer in `data_of()`. An op you want to keep is `aiednn_gop_wildcard`; `run_host` is where it runs, using the pointers the partition reports.

`dnn_build` chooses kernels by scanning this file for primitive create calls. The comment at the top names `aiednn_matmul_forward_create(` so the matmul partition has a kernel to compile to. A convolution you add needs its create named the same way.
