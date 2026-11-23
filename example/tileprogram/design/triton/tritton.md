<!-- Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
     SPDX-License-Identifier: Apache-2.0 -->
# Triton-style Python Frontend for AIE — Design Document

## Overview

This document describes the design of a Triton-style Python frontend for programming AMD Versal AI Engine (AIE) via aiehlc. The goal is to let users write AIE tile kernels using familiar Triton syntax (`@triton.jit`, `tl.load`, `tl.dot`, `tl.store`) while the compiler handles all AIE-specific concerns: DMA transfers, stream switch routing, buffer descriptors, lock synchronization, and core ELF generation.

Reference implementation: `triton_matmul.py` (Python) and `proposal3.cc` (C equivalent).

---

## Compilation Flow

```
triton_matmul.py
  │  Python AST parse (@aie_triton.jit functions)
  ▼
routing dialect IR
  │  buildRoutingIR(ctx, meshRows, meshCols, tensors)
  │  (tilinglinalg_pipeline.h → TensorParam per tensor)
  ▼
MLIR lowering pipeline
  │  routing → dmap → dmaphop → dfscheblueprint → dfschedule
  ▼
host.cc + kernel.cc + routing.cc + aieml.bcf + aieml.prx
  │  cross-compile: xchesscc (kernel ELF), aarch64-g++ (host ELF)
  ▼
Deploy on AIE HW: load ELFs, configure DMA BDs, run cores
```

### Stages

| Stage | Input | Output | Implementation |
|-------|-------|--------|----------------|
| 1. Python AST parse | `.py` with `@aie_triton.jit` | Kernel body + tensor metadata | Python `ast` module |
| 2. Build routing IR | `TensorParam[]` + mesh dims | `mlir::ModuleOp` (routing dialect) | `TilingLinalgPipeline::buildRoutingIR()` |
| 3. MLIR pipeline | routing IR | host.cc, kernel.cc, routing.cc | `TilingLinalgPipeline::runPipeline()` |
| 4. Cross-compile | .cc sources + .bcf/.prx | kernel ELF + host ELF | xchesscc, aarch64-g++ |
| 5. Deploy | ELF binaries | Running on AIE HW | XAie driver APIs |

---

## Concept Mapping: Triton Python to AIE C

Every Triton construct has a direct mapping to the AIE C programming model used in `proposal3.cc`.

### Kernel Declaration

| Triton Python | AIE C (proposal3.cc) |
|---------------|----------------------|
| `@aie_triton.jit` | `__global__` |
| `def matmul(a_ptr, b_ptr, c_ptr, ...)` | `void matmul(input_window_int8 *window_in_0, input_window_int8 *window_in_1, output_window_int8 *window_out_0)` |

The Python frontend rewrites pointer-based arguments into window arguments. `a_ptr` and `b_ptr` (used with `tl.load`) become `input_window_int8 *`, while `c_ptr` (used with `tl.store`) becomes `output_window_int8 *`.

### Tile Identity

| Triton Python | AIE C (proposal3.cc) |
|---------------|----------------------|
| `tile_row = tl.program_id(axis=0)` | `int row = coreid & 0x1F;` |
| `tile_col = tl.program_id(axis=1)` | `int col = coreid >> 16;` |

On GPU Triton, `program_id` returns a 1D block index that is manually decomposed into M/N indices. On AIE, the mesh is inherently 2D — `program_id(0)` maps to the physical tile row and `program_id(1)` to the column, extracted from `get_coreid()`.

### Data Movement

| Triton Python | AIE C (proposal3.cc) | Notes |
|---------------|----------------------|-------|
| `a_block = tl.load(a_ptr + offsets)` | `int8_t *in0 = (int8_t *)acquire_input_window(window_in_0);` | DMA fetches data from DDR into tile-local ping/pong buffer; lock acquire waits until ready |
| `b_block = tl.load(b_ptr + offsets)` | `int8_t *in1 = (int8_t *)acquire_input_window(window_in_1);` | Same mechanism for second input |
| `tl.store(c_ptr + offsets, result)` | `int8_t *out = acquire_output_window(window_out_0);` then write, then `release_output_window(window_out_0);` | DMA writes from tile-local buffer back to DDR |
| (end of K-iteration scope) | `release_input_window(window_in_0);` `release_input_window(window_in_1);` | Compiler inserts lock release at DMA boundary |

### Compute

| Triton Python | AIE C (proposal3.cc) | Notes |
|---------------|----------------------|-------|
| `acc = tl.zeros((8,8), dtype=tl.int32)` | Local buffer at BCF-assigned address | Tile-local memory, ~32 KB available |
| `acc += tl.dot(a_block, b_block)` | `v4int8 data0 = *((v4int8 *)&in0[i*4]); ...` MAC loop | AIE vector multiply-accumulate (v4int8 or v16int8) |
| `result = acc.to(tl.int8)` | Saturate int32 accumulator to int8 | Done in kernel before output store |

### Host API

| Triton Python | AIE C (proposal3.cc) |
|---------------|----------------------|
| `aie_triton.set_device(0)` | `aieSetDevice(0);` |
| `mesh = aie_triton.mesh(rows=2, cols=2)` | `aieDim mesh(2, 2);` |
| `A = np.arange(1, M*K+1, dtype=np.int8)` | `int32_t *A = malloc(...); for(i) A[i]=i+1;` |
| `matmul_simple[grid](A, B, C, M, N, K, ...)` | `matmul<<<mesh>>>(A, B, C, M, N, K);` |
| `aie_triton.synchronize()` | `aieDeviceSynchronize();` |

---

## Kernel Variants

`triton_matmul.py` provides two kernel variants. Both compile to the same routing IR.

### Variant 1: Simple / Window API (`matmul_simple`)

Closest to `proposal3.cc`. Uses offset-based `tl.load`/`tl.store` with fixed 8x8 blocks:

```python
@aie_triton.jit
def matmul_simple(a_ptr, b_ptr, c_ptr, M, N, K,
                  BLOCK_M: tl.constexpr = 8,
                  BLOCK_N: tl.constexpr = 8,
                  BLOCK_K: tl.constexpr = 8):
    tile_row = tl.program_id(axis=0)
    tile_col = tl.program_id(axis=1)
    row_start = tile_row * BLOCK_M
    col_start = tile_col * BLOCK_N
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)

    for k_start in range(0, K, BLOCK_K):
        a_block = tl.load(a_ptr + ...)  # → acquire_input_window(window_in_0)
        b_block = tl.load(b_ptr + ...)  # → acquire_input_window(window_in_1)
        acc += tl.dot(a_block, b_block) # → v4int8 MAC

    result = acc.to(tl.int8)
    tl.store(c_ptr + ..., result)       # → acquire/write/release output window
```

Direct correspondence to `proposal3.cc`:

```c
__global__ void matmul(input_window_int8 *window_in_0,
                       input_window_int8 *window_in_1,
                       output_window_int8 *window_out_0) {
    for (int k = 0; k < 2; k++) {
        int8_t *in0 = (int8_t *)acquire_input_window(window_in_0);
        int8_t *in1 = (int8_t *)acquire_input_window(window_in_1);
        int8_t *out = acquire_output_window(window_out_0);
        for (int i = 0; i < BUF_SZ; i++) {
            v4int8 data0 = *((v4int8 *)&in0[i * 4]);
            v4int8 data1 = *((v4int8 *)&in1[i * 4]);
            *((v4int8 *)&out[i * 4]) = data0; // placeholder
        }
        release_input_window(window_in_0);
        release_input_window(window_in_1);
        release_output_window(window_out_0);
    }
}
```

### Variant 2: Strided / Triton-standard (`matmul_strided`)

Standard Triton matmul tutorial pattern with `tl.make_block_ptr` and `tl.advance`, for users coming from GPU Triton:

```python
@aie_triton.jit
def matmul_strided(a_ptr, b_ptr, c_ptr, M, N, K,
                   stride_am, stride_ak, stride_bk, stride_bn,
                   stride_cm, stride_cn,
                   BLOCK_M: tl.constexpr = 8, ...):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    a_block_ptr = tl.make_block_ptr(base=a_ptr, shape=(M,K),
        strides=(stride_am, stride_ak),
        offsets=(pid_m*BLOCK_M, 0), block_shape=(BLOCK_M, BLOCK_K), ...)

    b_block_ptr = tl.make_block_ptr(base=b_ptr, shape=(K,N),
        strides=(stride_bk, stride_bn),
        offsets=(0, pid_n*BLOCK_N), block_shape=(BLOCK_K, BLOCK_N), ...)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
    for k in range(0, K, BLOCK_K):
        acc += tl.dot(tl.load(a_block_ptr), tl.load(b_block_ptr))
        a_block_ptr = tl.advance(a_block_ptr, (0, BLOCK_K))
        b_block_ptr = tl.advance(b_block_ptr, (BLOCK_K, 0))

    c_block_ptr = tl.make_block_ptr(...)
    tl.store(c_block_ptr, acc.to(tl.int8))
```

The Python frontend normalizes both variants into identical routing IR via `buildRoutingIR()`.

---

## Data Partitioning

For a 16x16 matmul on a 2x2 tile mesh with BLOCK_M=8, BLOCK_N=8, BLOCK_K=8:

```
C = A @ B          A: [16,16]   B: [16,16]   C: [16,16]

        col 0       col 1
       ┌───────┬───────┐
row 0  │(0,0)  │(0,1)  │    tile(0,0) → C[0:8,  0:8 ]
       │C[0:8, │C[0:8, │    tile(0,1) → C[0:8,  8:16]
       │  0:8] │ 8:16] │    tile(1,0) → C[8:16, 0:8 ]
       ├───────┼───────┤    tile(1,1) → C[8:16, 8:16]
row 1  │(1,0)  │(1,1)  │
       │C[8:16,│C[8:16,│
       │  0:8] │ 8:16] │
       └───────┴───────┘
```

Per-tile data requirements:

| Tensor | Per-tile shape | Streamed? | DMA chunks |
|--------|---------------|-----------|------------|
| A partition | 8 x 16 (full K, partial M) | Yes, in K-dim | 2 chunks of 8x8 |
| B partition | 16 x 8 (full K, partial N) | Yes, in K-dim | 2 chunks of 8x8 |
| C partition | 8 x 8 | No (written once) | 1 chunk of 8x8 |

Each 8x8 int8 block = 64 bytes. Total tile-local memory per iteration: 64 (A) + 64 (B) + 64 (C) + 256 (acc, int32) = 448 bytes, well within the 32 KB limit. With ping-pong buffering, double this to ~896 bytes.

---

## buildRoutingIR Integration

The Python frontend extracts metadata from the `@aie_triton.jit` decorated function and the `kernel[grid](...)` call site to construct the parameters for `buildRoutingIR()` (defined in `tilinglinalg_pipeline.h`):

### TensorParam Extraction

```
Python source                           → TensorParam
─────────────────────────────────────────────────────────
a_ptr used in tl.load(a_ptr + ...)      → TensorParam{shape={16,16}, elementBitWidth=8, isInput=true }
b_ptr used in tl.load(b_ptr + ...)      → TensorParam{shape={16,16}, elementBitWidth=8, isInput=true }
c_ptr used in tl.store(c_ptr + ...)     → TensorParam{shape={16,16}, elementBitWidth=8, isInput=false}
```

Extraction rules:
- **shape**: Inferred from the numpy array passed at the call site (`A.shape` = `(16,16)`)
- **elementBitWidth**: Inferred from numpy dtype (`np.int8` → 8)
- **isInput**: `tl.load` targets → `true`; `tl.store` targets → `false`

### API Call

```cpp
// Constructed by the Python frontend:
std::vector<TensorParam> tensors = {
    {{16, 16}, 8, true },   // A
    {{16, 16}, 8, true },   // B
    {{16, 16}, 8, false},   // C
};

mlir::ModuleOp module = TilingLinalgPipeline::buildRoutingIR(ctx, 2, 2, tensors);
//                                                           meshRows meshCols

bool ok = TilingLinalgPipeline::runPipeline(ctx, module, outputDir,
    userKernelBody,       // extracted from @aie_triton.jit function body
    userKernelFuncName);  // "matmul_simple" or "matmul_strided"
```

### Mesh → Grid Mapping

```python
mesh = aie_triton.mesh(rows=2, cols=2)    # → meshRows=2, meshCols=2
grid = (mesh.rows, mesh.cols)              # → (2, 2)
matmul_simple[grid](A, B, C, ...)         # launches 2x2 = 4 tile programs
```

Each `(row, col)` in the grid is a physical AIE tile. Unlike GPU Triton where the grid is 1D and programs are scheduled dynamically, AIE programs have a fixed 1:1 mapping to physical tiles.

---

## AIE Constraints vs GPU Triton

| Aspect | GPU Triton | AIE Triton |
|--------|-----------|------------|
| **Grid** | 1D/2D/3D, dynamically scheduled | 2D mesh, statically mapped to physical tiles |
| **Memory per program** | Shared memory: 48-228 KB | Tile-local data memory: ~32 KB |
| **Block sizes** | Typically 64-256 | Typically 4-16 (int8) |
| **Data types** | fp16, fp32, bf16, int8 | int8, int16, int32 (AIEML native) |
| **Data movement** | Global → shared memory (software managed) | DDR → tile DMA (compiler managed, lock-synchronized) |
| **Masking** | Required for boundary conditions | Not needed when dims are exact multiples of block size |
| **Pointer arithmetic** | Real GPU pointers | Logical — frontend rewrites to window acquire/release |
| **K-loop iterations** | Many (large matrices) | Few (small partitions, e.g., 2 for K=16, BLOCK_K=8) |
| **Ping-pong** | Software-managed shared memory staging | Hardware DMA ping-pong buffers with lock handshake |

---

## Host Code Structure

The host section in `triton_matmul.py` matches `proposal3.cc`'s `main()` line by line:

```
proposal3.cc                              triton_matmul.py
────────────────────────────────────────  ────────────────────────────────────────
const int M=16, N=16, K=16;              M, N, K = 16, 16, 16
aieSetDevice(0);                         aie_triton.set_device(0)
aieDim mesh(2, 2);                       mesh = aie_triton.mesh(rows=2, cols=2)
int32_t *A = malloc(M*K*sizeof);         A = np.arange(1, M*K+1, dtype=np.int8).reshape(M,K)
for(i) A[i] = i+1;                      (done by np.arange)
int32_t *B = malloc(K*N*sizeof);         B = np.arange(1, K*N+1, dtype=np.int8).reshape(K,N)
for(i) B[i] = i+1;                      (done by np.arange)
int32_t *C = malloc(M*N*sizeof);         C = np.zeros((M,N), dtype=np.int8)
matmul<<<mesh>>>(A,B,C,M,N,K);          matmul_simple[grid](A,B,C,M,N,K,...)
aieDeviceSynchronize();                  aie_triton.synchronize()
verify loop with printf                  numpy comparison + print
free(A); free(B); free(C);              (garbage collected)
```

---

## aie_triton Module API

The `aie_triton` module exposes these functions (mirroring Triton's `triton` module):

| Function | Purpose | Maps to |
|----------|---------|---------|
| `@aie_triton.jit` | Decorator marking an AIE tile kernel | `__global__` in C |
| `aie_triton.set_device(id)` | Select AIE device | `aieSetDevice(id)` |
| `aie_triton.mesh(rows, cols)` | Define tile mesh dimensions | `aieDim mesh(rows, cols)` |
| `aie_triton.synchronize()` | Wait for all tiles to complete | `aieDeviceSynchronize()` |

The `aie_triton.language` module (imported as `tl`) exposes:

| Function | Purpose | Maps to |
|----------|---------|---------|
| `tl.program_id(axis)` | Get tile row (0) or col (1) | `get_coreid()` bit extraction |
| `tl.load(ptr + offsets)` | Load block from DDR via DMA | `acquire_input_window()` |
| `tl.store(ptr + offsets, val)` | Store block to DDR via DMA | `acquire_output_window()` + write + `release_output_window()` |
| `tl.dot(a, b)` | Matrix multiply-accumulate | `v4int8` / `v16int8` MAC loop |
| `tl.zeros(shape, dtype)` | Allocate zero-initialized local buffer | BCF-assigned tile-local address |
| `tl.arange(start, end)` | Index range for offset computation | (compiled away — DMA BDs encode offsets) |
| `tl.constexpr` | Compile-time constant annotation | Becomes template parameter |
| `tl.make_block_ptr(...)` | Structured block pointer (Triton 2.0+) | Frontend rewrites to window + DMA region |
| `tl.advance(ptr, offsets)` | Advance block pointer along dims | Next DMA BD / ping-pong swap |
| `val.to(dtype)` | Type cast (e.g., int32 → int8 saturate) | In-kernel type conversion |

---

## File Inventory

| File | Role |
|------|------|
| `triton_matmul.py` | Triton-style Python input (this design's reference implementation) |
| `proposal3.cc` | C equivalent using `__global__` + window API |
| `tilinglinalg_pipeline.h` | `TensorParam` struct + `buildRoutingIR()` / `runPipeline()` API |

---

## Design Decisions

1. **Two kernel variants in one file** — `matmul_simple` gives the clearest 1:1 mapping to `proposal3.cc` for understanding the AIE programming model. `matmul_strided` provides the standard Triton idiom for users migrating from GPU code. Both compile to identical routing IR.

2. **int8 data type** — Matches AIEML native vector width and `proposal3.cc`'s `input_window_int8` / `v4int8` types. The accumulator uses int32 to avoid overflow, with saturation to int8 on output.

3. **8x8 block sizes** — Chosen to fit within 32 KB tile-local memory with ping-pong buffering. For the 16x16 matmul on a 2x2 mesh, this gives exactly 2 K-iterations, matching `proposal3.cc`'s `for (int k = 0; k < 2; k++)`.

4. **2D program_id** — Unlike GPU Triton's 1D `program_id(0)` that requires manual decomposition, AIE's physical 2D mesh maps naturally to `program_id(axis=0)` for row and `program_id(axis=1)` for column.

5. **Implicit lock/release** — In `proposal3.cc`, `release_input_window` / `release_output_window` are explicit. In the Triton syntax, these are implicit — the compiler inserts them at DMA boundaries based on `tl.load` / `tl.store` lifetime analysis.

6. **Host uses numpy** — numpy arrays replace C `malloc` + init loops. The `np.arange(1, M*K+1)` pattern exactly reproduces `for(i) A[i] = i+1`.
