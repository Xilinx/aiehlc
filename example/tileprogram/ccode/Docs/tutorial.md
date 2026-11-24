<!-- Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
     SPDX-License-Identifier: Apache-2.0 -->
# AIEHLC Tile Programming Tutorial

Write C/C++. Run it on AIE tiles. That's the idea.

AIEHLC gives you a RCOM/CUDA style programming model for AMD Versal AI Engine.
You write one `.cc` file containing **host code** (runs on ARM) and **kernel code** (runs on AIE tiles).
The compiler handles everything in between: DMA transfers, routing, buffer descriptors, locks, core loading.

---

## Design Philosophy

- **High Level abstraction on data movement** -- DMA, routing, locks, and ping-pong buffering are fully compiler-generated. You just pass pointers.
- **Low Level abstraction on computation** -- Inside `__global__`, you write plain C. You control every loop, every multiply, every optimization.
- **Progressive lowering** -- 6 MLIR dialects (routing &rarr; dmap &rarr; dmaphop &rarr; blueprint &rarr; dfschedule &rarr; EmitC) give you quick defaults and the flexibility to tune at any layer.

---

## The Programming Model in 30 Seconds

```
 You write                       The compiler generates
 ─────────                       ──────────────────────
 __global__ void kernel(...)     → kernel ELF for each AIE core
 main() { kernel<<<mesh>>>(); }  → host ELF with DMA + routing setup
```

A program has two parts:

| Part | Marked by | Runs on | What it does |
|------|-----------|---------|-------------|
| **Kernel** | `__global__` | AIE tile cores | Compute (matmul, conv, etc.) |
| **Host** | `main()` | ARM CPU | Allocate memory, launch kernels, verify results |

---

## Minimal Example: Vector Add

The smallest possible AIEHLC program:

```cpp
#include <stdint.h>

#define N 64

// ── Kernel: runs on every AIE tile ──
__global__ void matmul(aie::row_broadcast_in<input_window_int8 *> win_a,
                       aie::col_broadcast_in<input_window_int8 *> win_b,
                       aie::row_major_out<output_window_int8 *> win_c) {
    for (int iter = 0; iter < 2; iter++) {      // ping-pong: 2 iterations
        int8_t *a   = (int8_t *)acquire_input_window(win_a);
        int8_t *b   = (int8_t *)acquire_input_window(win_b);
        int8_t *out = acquire_output_window(win_c);

        for (int i = 0; i < N; i++)
            out[i] = a[i] * b[i];

        release_input_window(win_a);
        release_input_window(win_b);
        release_output_window(win_c);
    }
}

// ── Host: runs on ARM CPU ──
int main() {
    int8_t *A = (int8_t *)malloc(N);
    int8_t *B = (int8_t *)malloc(N);
    int8_t *C = (int8_t *)malloc(N);

    // Initialize A and B ...

    aieDim mesh(2, 2);             // 2x2 tile mesh
    matmul<<<mesh>>>(A, B, C);     // launch kernel
    aieDeviceSynchronize();        // wait for tiles to finish

    // Verify C ...

    free(A); free(B); free(C);
    return 0;
}
```

That's a complete program. The compiler will:
1. Partition A, B across the 2x2 mesh
2. Generate DMA buffer descriptors to move data to/from tiles
3. Set up stream switch routing
4. Configure locks for producer-consumer synchronization
5. Compile the kernel for AIE cores, the host for ARM

---

## API Reference

The entire API fits on one page.

### Host-Side

| API | Purpose |
|-----|---------|
| `aieSetDevice(0)` | Select AIE device (optional, defaults to 0) |
| `aieDim mesh(rows, cols)` | Declare tile mesh dimensions |
| `kernel<<<mesh>>>(A, B, C)` | Launch kernel across all tiles in mesh |
| `aieDeviceSynchronize()` | Block until all tiles finish |

Host code is standard C/C++. Use `malloc`/`free` for DDR memory.
Pointers passed to `kernel<<<mesh>>>()` are automatically partitioned and transferred to tiles via DMA.

### Kernel-Side

| API | Purpose |
|-----|---------|
| `__global__ void name(...)` | Declare a kernel function |
| `acquire_input_window(win)` | Get pointer to input data (blocks until DMA delivers it) |
| `acquire_output_window(win)` | Get pointer to output buffer (blocks until buffer is free) |
| `release_input_window(win)` | Signal that input has been consumed |
| `release_output_window(win)` | Signal that output has been written |
| `get_coreid()` | Get hardware tile ID (col in bits 16+, row in bits 0-4) |
| `klog(tag, value)` | Print a debug message from the tile |

### Window Types

Kernel parameters must use these types:

| Type | Description |
|------|-------------|
| `input_window_int8 *` | Input buffer of `int8_t` values |
| `output_window_int8 *` | Output buffer of `int8_t` values |

### Data Flow Pattern

Every kernel follows the same structure: a **ping-pong loop** wrapping the **acquire-compute-release** pattern.

```cpp
__global__ void my_kernel(input_window_int8 *in, output_window_int8 *out) {
    for (int iter = 0; iter < 2; iter++) {                  // ping-pong loop
        int8_t *input  = (int8_t *)acquire_input_window(in);   // 1. acquire
        int8_t *output = acquire_output_window(out);

        // 2. compute
        for (int i = 0; i < SIZE; i++)
            output[i] = input[i] * 2;

        release_input_window(in);                               // 3. release
        release_output_window(out);
    }
}
```

**Why `for (iter = 0; iter < 2; ...)`?** The compiler generates DMA buffer descriptors using ping-pong double buffering — two physical buffers alternate so that DMA can fill one while the kernel processes the other. The kernel must iterate twice to match this hardware protocol. Both iterations perform the same computation on the same data; the loop exists to stay synchronized with the DMA engine.

The acquire/release calls synchronize with DMA hardware locks.
**Acquire** blocks until data is ready. **Release** signals that the buffer can be reused.

---

## Example: Matrix Multiply

A realistic GEMM kernel: C[16x16] = A[16x16] * B[16x16], deployed on a 2x2 tile mesh.

```cpp
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#define M 16
#define K 16
#define N 16
#define M_TILE (M / 2)   // 8 rows per tile (2 tile rows)

__global__ void matmul(input_window_int8 *win_a,
                       input_window_int8 *win_b,
                       output_window_int8 *win_c) {
    for (int iter = 0; iter < 2; iter++) {
        int8_t *A = (int8_t *)acquire_input_window(win_a);
        int8_t *B = (int8_t *)acquire_input_window(win_b);
        int8_t *C = acquire_output_window(win_c);

        // C_tile[i][j] = sum_k( A_tile[i][k] * B[k][j] )
        for (int i = 0; i < M_TILE; i++) {
            for (int j = 0; j < N; j++) {
                int16_t sum = 0;
                for (int k = 0; k < K; k++)
                    sum += (int16_t)A[i * K + k] * (int16_t)B[k * N + j];
                if (sum > 127)       sum = 127;
                else if (sum < -128) sum = -128;
                C[i * N + j] = (int8_t)sum;
            }
        }

        release_input_window(win_a);
        release_input_window(win_b);
        release_output_window(win_c);
    }
}

int main() {
    aieSetDevice(0);
    aieDim mesh(2, 2);

    int8_t *A = (int8_t *)malloc(M * K);
    int8_t *B = (int8_t *)malloc(K * N);
    int8_t *C = (int8_t *)malloc(M * N);

    for (int i = 0; i < M * K; i++) A[i] = (int8_t)((i % 7) - 3);
    for (int i = 0; i < K * N; i++) B[i] = (int8_t)((i % 5) - 2);
    for (int i = 0; i < M * N; i++) C[i] = 0;

    matmul<<<mesh>>>(A, B, C);
    aieDeviceSynchronize();

    // Verify against CPU reference ...

    free(A); free(B); free(C);
    return 0;
}
```

**What happens under the hood:**

```
 Host DDR Memory                AIE 2x2 Tile Mesh
 ┌──────────────┐    DMA       ┌────────┬────────┐
 │ A[16x16]     │───────────>  │ Tile   │ Tile   │  Row 0: A[0:7,:], B[all]
 │ B[16x16]     │              │ (0,0)  │ (0,1)  │  → computes C[0:7,:]
 │              │              ├────────┼────────┤
 │ C[16x16]     │<───────────  │ Tile   │ Tile   │  Row 1: A[8:15,:], B[all]
 │              │              │ (1,0)  │ (1,1)  │  → computes C[8:15,:]
 └──────────────┘              └────────┴────────┘
```

The compiler partitions A by rows across tile rows and broadcasts B to all tiles.
Each tile computes its slice of C. The host sees a single contiguous C array.

---

## Chaining Multiple Kernels **WIP**

Kernels can be chained. The output of one becomes the input of the next.
This example shows two fused Conv+BN+ReLU layers back-to-back:

```cpp
__global__ void conv_bn_relu_1(input_window_int8 *feat_in,
                                input_window_int8 *params,
                                output_window_int8 *feat_out) {
    for (int iter = 0; iter < 2; iter++) {
        int8_t *in  = (int8_t *)acquire_input_window(feat_in);
        int8_t *p   = (int8_t *)acquire_input_window(params);
        int8_t *out = acquire_output_window(feat_out);
        // ... conv + batchnorm + leaky_relu computation ...
        release_input_window(feat_in);
        release_input_window(params);
        release_output_window(feat_out);
    }
}

__global__ void conv_bn_relu_2(input_window_int8 *feat_in,
                                input_window_int8 *params,
                                output_window_int8 *feat_out) {
    // ... same pattern with for(2) ping-pong loop, different weights ...
}

int main() {
    aieDim mesh(2, 2);

    int8_t *input        = (int8_t *)malloc(64);
    int8_t *params1      = (int8_t *)malloc(11);   // 3x3 filter + BN scale + bias
    int8_t *params2      = (int8_t *)malloc(11);
    int8_t *intermediate = (int8_t *)malloc(64);
    int8_t *output       = (int8_t *)malloc(64);

    // Layer 1: input → intermediate
    conv_bn_relu_1<<<mesh>>>(input, params1, intermediate);

    // Layer 2: intermediate → output
    conv_bn_relu_2<<<mesh>>>(intermediate, params2, output);

    aieDeviceSynchronize();

    // Verify ...
}
```

Each `<<<mesh>>>` launch is an independent kernel dispatch.
The compiler handles the intermediate buffer routing between layers.

---

## Building a Full Pipeline (Self-Attention) **WIP**

For complex models, compose multiple kernel types into a dataflow graph:

```cpp
// 5 different kernel types (each has for(2) ping-pong loop inside)
__global__ void linear_proj(input_window_int8 *in, input_window_int8 *w,
                             output_window_int8 *out) {
    for (int iter = 0; iter < 2; iter++) { /* acquire → matmul → release */ }
}
__global__ void matmul_qk(input_window_int8 *q, input_window_int8 *k,
                           output_window_int8 *scores) {
    for (int iter = 0; iter < 2; iter++) { /* acquire → Q*K^T/sqrt(d) → release */ }
}
__global__ void approx_softmax(input_window_int8 *scores,
                                output_window_int8 *attn) {
    for (int iter = 0; iter < 2; iter++) { /* acquire → softmax → release */ }
}
__global__ void attn_apply(input_window_int8 *attn, input_window_int8 *v,
                            output_window_int8 *context) {
    for (int iter = 0; iter < 2; iter++) { /* acquire → attn * V → release */ }
}
__global__ void residual_add(input_window_int8 *a, input_window_int8 *b,
                              output_window_int8 *out) {
    for (int iter = 0; iter < 2; iter++) { /* acquire → a + b → release */ }
}

int main() {
    aieDim mesh(2, 2);

    // Allocate all buffers ...

    // Forward pass: a sequence of kernel launches
    linear_proj<<<mesh>>>(input, Wq, Q);        // Q = input * Wq
    linear_proj<<<mesh>>>(input, Wk, K);        // K = input * Wk
    linear_proj<<<mesh>>>(input, Wv, V);        // V = input * Wv
    matmul_qk<<<mesh>>>(Q, K, scores);          // scores = Q * K^T / sqrt(d)
    approx_softmax<<<mesh>>>(scores, attn);     // attn = softmax(scores)
    attn_apply<<<mesh>>>(attn, V, context);     // context = attn * V
    linear_proj<<<mesh>>>(context, Wo, proj);   // proj = context * Wo
    residual_add<<<mesh>>>(input, proj, result); // result = input + proj

    aieDeviceSynchronize();
}
```

Each line is a kernel launch. The compiler schedules them, routes data between tiles, and manages all DMA transfers. You just describe the computation.

---

## Debug Support

### Debug Level Pragma

Control kernel debug verbosity:

```cpp
#pragma aie_debug_level 2
__global__ void my_kernel(...) { ... }
```

### Tile ID Inspection

Use `get_coreid()` to identify which tile is executing:

```cpp
__global__ void my_kernel(input_window_int8 *in, output_window_int8 *out) {
    unsigned coreid = get_coreid();
    int col = coreid >> 16;
    int row = coreid & 0x1F;
    klog("TILE", row * 10 + col);  // e.g., tile (1,2) logs 12

    // ... normal computation ...
}
```

### Kernel Logging

`klog(tag, value)` emits a tagged debug message from inside the kernel.
Messages are captured in the host console output.

```cpp
klog("CENk", iter);                         // entering kernel, iteration N
klog("IN0", (int8_t)(uintptr_t)ptr);        // log buffer address
klog("CLOP", num_elements);                 // computation done
klog("CEXT", 1);                            // kernel exiting
```

---

## Build and Run

### Step 1: Clone the repository

```bash
git clone <repo-url> aiehlc
cd aiehlc
```

### Step 2: Set up the environment

`setup.sh` detects and configures Vitis, cross-compiler toolchains, the aie-rt driver, and build directories.

```bash
source script/setup.sh
```

If your aie-rt BSP is in a git repository rather than the default location:

```bash
source script/setup.sh --bsp-use-git-repo=https://path/to/aie-rt.git
```

### Step 3: Compile your program

`aiehlc.sh` takes a single `.cc` source file and produces a host ELF (ARM) and kernel ELF (AIE core). It handles Clang AST parsing, MLIR lowering, kernel compilation (via xchesscc), and host cross-compilation in one command.

```bash
source script/aiehlc.sh --runtime-source-file your_program.cc
```

Alternative compiler backends:

```bash
# Using LLVM-AIE (experimental, open-source kernel compiler)
source script/aiehlc.sh --use-llvm-aie --runtime-source-file your_program.cc

# Targeting AIE2PS devices
source script/aiehlc.sh --aie-version 5 --runtime-source-file your_program.cc

# PetaLinux (Linux on ARM instead of baremetal)
source script/aiehlc.sh --platform linux --aie-version 2 --runtime-source-file your_program.cc
```

### Step 4: Run on hardware

```bash
python3 script/test/apppaltest.py build/host
```

This script handles SSH + xsdb board connection, ELF loading, console capture, and result verification.

### Optional: Build aiehlc itself

Only needed if you are developing the compiler. See [doc/build.md](../../../../doc/build.md) for details.

```bash
mkdir build && cd build
cmake .. -DLLVM_INSTALL_DIR=/path/to/llvm/build
make -j$(nproc)
```

> For full details on setup options, debug levels, llvm-aie support, and PetaLinux compilation, see the root [README.md](../../../../README.md).

---

## Summary: What You Write vs. What the Compiler Does

| You write | The compiler handles |
|-----------|---------------------|
| `__global__ void kernel(...)` | Compiling kernel for AIE cores (via xchesscc) |
| `acquire/release` window calls | Lock acquire/release sequences |
| `kernel<<<mesh>>>(A, B, C)` | DMA buffer descriptors, data partitioning |
| `aieDim mesh(R, C)` | Stream switch routing across R*C tiles |
| `aieDeviceSynchronize()` | Core load, run, wait, teardown |
| `malloc` / `free` | DDR memory mapping for GMIO transfers |

You write the math. The compiler builds the machine.
