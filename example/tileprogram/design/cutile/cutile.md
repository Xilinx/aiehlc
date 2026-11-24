<!-- Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
     SPDX-License-Identifier: Apache-2.0 -->
# CuTile-style Python Frontend for AIE — Design Document

## Overview

This document describes the design of a CuTile-style Python frontend for programming AMD Versal AI Engine (AIE) via aiehlc. The goal is to let users write AIE tile kernels using NVIDIA CuTile API conventions (`import cuda.tile as ct`, `@ct.kernel`, `ct.load`, `ct.mma`, `ct.store`) while the compiler handles all AIE-specific concerns: DMA transfers, stream switch routing, buffer descriptors, lock synchronization, and core ELF generation.

CuTile (NVIDIA CUDA Tile API, CUDA 13.1+) uses **index-based tile addressing** rather than Triton's pointer arithmetic. This maps more naturally to AIE's DMA buffer descriptor model, where transfers are specified by tile index and shape rather than base pointer + offset computation.

Reference implementations: `cutile_matmul.py` (CuTile Python), `triton_matmul.py` (Triton Python), and `proposal3.cc` (C equivalent).

---

## Compilation Flow

```
cutile_matmul.py
  │  Python AST parse (@ct.kernel functions)
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
| 1. Python AST parse | `.py` with `@ct.kernel` | Kernel body + tensor metadata | Python `ast` module |
| 2. Build routing IR | `TensorParam[]` + mesh dims | `mlir::ModuleOp` (routing dialect) | `TilingLinalgPipeline::buildRoutingIR()` |
| 3. MLIR pipeline | routing IR | host.cc, kernel.cc, routing.cc | `TilingLinalgPipeline::runPipeline()` |
| 4. Cross-compile | .cc sources + .bcf/.prx | kernel ELF + host ELF | xchesscc, aarch64-g++ |
| 5. Deploy | ELF binaries | Running on AIE HW | XAie driver APIs |

---

## Three-way Concept Mapping: CuTile ←→ Triton ←→ AIE C

Every CuTile construct has a direct mapping to both the Triton and AIE C programming models.

### Kernel Declaration

| CuTile Python | Triton Python | AIE C (proposal3.cc) |
|---------------|---------------|----------------------|
| `@ct.kernel` | `@aie_triton.jit` | `__global__` |
| `def matmul(A: ct.Tensor, B: ct.Tensor, C: ct.Tensor, ...)` | `def matmul(a_ptr, b_ptr, c_ptr, ...)` | `void matmul(input_window_int8 *window_in_0, input_window_int8 *window_in_1, output_window_int8 *window_out_0)` |

CuTile uses typed `ct.Tensor` parameters; Triton uses raw pointers; AIE C uses typed window parameters. The Python frontend rewrites all three into the same window arguments.

### Tile Identity

| CuTile Python | Triton Python | AIE C (proposal3.cc) |
|---------------|---------------|----------------------|
| `bidx = ct.program_id(0)` | `tile_row = tl.program_id(axis=0)` | `int row = coreid & 0x1F;` |
| `bidy = ct.program_id(1)` | `tile_col = tl.program_id(axis=1)` | `int col = coreid >> 16;` |

On GPU CuTile, `program_id` returns a block index within the grid. On AIE, the mesh is inherently 2D — `program_id(0)` maps to the physical tile row and `program_id(1)` to the column, extracted from `get_coreid()`.

### Data Movement

| CuTile Python | Triton Python | AIE C (proposal3.cc) | Notes |
|---------------|---------------|----------------------|-------|
| `a = ct.load(A, index=(bidx, k), shape=(tm, tk))` | `a_block = tl.load(a_ptr + offsets)` | `int8_t *in0 = (int8_t *)acquire_input_window(window_in_0);` | DMA fetches data from DDR into tile-local ping/pong buffer |
| `b = ct.load(B, index=(k, bidy), shape=(tk, tn))` | `b_block = tl.load(b_ptr + offsets)` | `int8_t *in1 = (int8_t *)acquire_input_window(window_in_1);` | Same mechanism for second input |
| `ct.store(C, index=(bidx, bidy), tile=result)` | `tl.store(c_ptr + offsets, result)` | `acquire_output_window` + write + `release_output_window` | DMA writes from tile-local buffer back to DDR |
| (end of K-iteration scope) | (implicit at `tl.load` lifetime end) | `release_input_window(window_in_0/1);` | Compiler inserts lock release at DMA boundary |

### Compute

| CuTile Python | Triton Python | AIE C (proposal3.cc) | Notes |
|---------------|---------------|----------------------|-------|
| `acc = ct.full((tm, tn), 0, dtype=ct.int32)` | `acc = tl.zeros((8,8), dtype=tl.int32)` | Local buffer at BCF-assigned address | Tile-local memory, ~32 KB available |
| `acc = ct.mma(a, b, acc)` | `acc += tl.dot(a_block, b_block)` | `v4int8 data0 = ...; data1 = ...; MAC` | AIE vector multiply-accumulate |
| `result = ct.astype(acc, ct.int8)` | `result = acc.to(tl.int8)` | Saturate int32 → int8 | Done in kernel before output store |

### K-loop Iteration

| CuTile Python | Triton Python | AIE C (proposal3.cc) |
|---------------|---------------|----------------------|
| `num_k = ct.num_tiles(A, axis=1, shape=(tm, tk))` | `range(0, K, BLOCK_K)` | `for (int k = 0; k < 2; k++)` |
| `for k in range(num_k):` | `for k_start in range(0, K, BLOCK_K):` | (same loop, count = K/BLOCK_K = 2) |

CuTile's `ct.num_tiles()` makes the iteration count explicit from the tensor and tile shape, matching AIE's fixed iteration count.

### Compile-time Constants

| CuTile Python | Triton Python | AIE C |
|---------------|---------------|-------|
| `tm: ct.Constant[int]` | `BLOCK_M: tl.constexpr = 8` | C template parameter / `#define` |

### Host API

| CuTile Python | Triton Python | AIE C (proposal3.cc) |
|---------------|---------------|----------------------|
| `ct.set_device(0)` | `aie_triton.set_device(0)` | `aieSetDevice(0);` |
| `mesh = ct.mesh(rows=2, cols=2)` | `mesh = aie_triton.mesh(rows=2, cols=2)` | `aieDim mesh(2, 2);` |
| `A = np.arange(1, M*K+1, dtype=np.int8)` | (same) | `int32_t *A = malloc(...); for(i) A[i]=i+1;` |
| `matmul_simple[grid](A, B, C, tm=8, tn=8, tk=8)` | `matmul_simple[grid](A, B, C, M, N, K, BLOCK_M=8, ...)` | `matmul<<<mesh>>>(A, B, C, M, N, K);` |
| `ct.synchronize()` | `aie_triton.synchronize()` | `aieDeviceSynchronize();` |

---

## Kernel Variants

`cutile_matmul.py` provides two kernel variants. Both compile to the same routing IR.

### Variant 1: Simple / Index-based (`matmul_simple`)

Closest to `proposal3.cc`. Uses `ct.load`/`ct.mma`/`ct.store` with index-based tile addressing:

```python
@ct.kernel
def matmul_simple(A: ct.Tensor, B: ct.Tensor, C: ct.Tensor,
                  tm: ct.Constant[int], tn: ct.Constant[int], tk: ct.Constant[int]):
    bidx = ct.program_id(0)                           # → get_coreid() & 0x1F
    bidy = ct.program_id(1)                           # → get_coreid() >> 16
    num_k = ct.num_tiles(A, axis=1, shape=(tm, tk))   # → 2
    acc = ct.full((tm, tn), 0, dtype=ct.int32)

    for k in range(num_k):
        a = ct.load(A, index=(bidx, k), shape=(tm, tk))   # → acquire_input_window
        b = ct.load(B, index=(k, bidy), shape=(tk, tn))   # → acquire_input_window
        acc = ct.mma(a, b, acc)                             # → v4int8 MAC

    result = ct.astype(acc, ct.int8)
    ct.store(C, index=(bidx, bidy), tile=result)           # → acquire/write/release
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

And to `triton_matmul.py`:

```python
@aie_triton.jit
def matmul_simple(a_ptr, b_ptr, c_ptr, M, N, K,
                  BLOCK_M: tl.constexpr = 8, ...):
    tile_row = tl.program_id(axis=0)
    tile_col = tl.program_id(axis=1)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)

    for k_start in range(0, K, BLOCK_K):
        a_block = tl.load(a_ptr + ...)  # → acquire_input_window
        b_block = tl.load(b_ptr + ...)  # → acquire_input_window
        acc += tl.dot(a_block, b_block) # → v4int8 MAC

    result = acc.to(tl.int8)
    tl.store(c_ptr + ..., result)       # → acquire/write/release
```

### Variant 2: With block swizzling (`matmul_swizzle`)

Standard CuTile pattern using `swizzle_2d` for GPU L2 cache locality. On AIE, the swizzle is a no-op since tiles are physically fixed, but it demonstrates the full CuTile API:

```python
@ct.kernel
def matmul_swizzle(A, B, C, tm, tn, tk, GROUP_SIZE_M: ct.Constant[int]):
    num_rows = ct.num_tiles(A, axis=0, shape=(tm, tk))
    bidx_sw, bidy_sw = swizzle_2d(ct.program_id(0), ct.program_id(1),
                                   num_rows, GROUP_SIZE_M)
    # ... same ct.load/ct.mma/ct.store loop as matmul_simple
```

The Python frontend normalizes both variants into identical routing IR via `buildRoutingIR()`.

Triton equivalent: the standard Triton matmul tutorial's `pid_m`/`pid_n` decomposition with `group_id`:
```python
pid = tl.program_id(0)
group_id = pid // (GROUP_SIZE_M * num_pid_n)
pid_m = first_pid_m + (pid % num_pid_in_group) % GROUP_SIZE_M
pid_n = (pid % num_pid_in_group) // GROUP_SIZE_M
```

---

## Data Partitioning

For a 16x16 matmul on a 2x2 tile mesh with tm=8, tn=8, tk=8:

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

| Tensor | Per-tile shape | Streamed? | DMA chunks | CuTile load pattern |
|--------|---------------|-----------|------------|---------------------|
| A partition | 8 x 16 (full K, partial M) | Yes, in K-dim | 2 chunks of 8x8 | `ct.load(A, index=(bidx, k), shape=(8,8))` |
| B partition | 16 x 8 (full K, partial N) | Yes, in K-dim | 2 chunks of 8x8 | `ct.load(B, index=(k, bidy), shape=(8,8))` |
| C partition | 8 x 8 | No (written once) | 1 chunk of 8x8 | `ct.store(C, index=(bidx, bidy), tile=result)` |

Each 8x8 int8 block = 64 bytes. Total tile-local memory per iteration: 64 (A) + 64 (B) + 64 (C) + 256 (acc, int32) = 448 bytes, well within the 32 KB limit. With ping-pong buffering, double this to ~896 bytes.

---

## buildRoutingIR Integration

The Python frontend extracts metadata from the `@ct.kernel` decorated function and the `kernel[grid](...)` call site to construct the parameters for `buildRoutingIR()` (defined in `tilinglinalg_pipeline.h`):

### TensorParam Extraction

```
Python source                                      → TensorParam
──────────────────────────────────────────────────────────────────
A: ct.Tensor used in ct.load(A, ...)               → TensorParam{shape={16,16}, elementBitWidth=8, isInput=true }
B: ct.Tensor used in ct.load(B, ...)               → TensorParam{shape={16,16}, elementBitWidth=8, isInput=true }
C: ct.Tensor used in ct.store(C, ...)              → TensorParam{shape={16,16}, elementBitWidth=8, isInput=false}
```

Extraction rules:
- **shape**: Inferred from the numpy array passed at the call site (`A.shape` = `(16,16)`)
- **elementBitWidth**: Inferred from numpy dtype (`np.int8` → 8)
- **isInput**: `ct.load` targets → `true`; `ct.store` targets → `false`

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
    userKernelBody,       // extracted from @ct.kernel function body
    userKernelFuncName);  // "matmul_simple" or "matmul_swizzle"
```

### Mesh → Grid Mapping

```python
mesh = ct.mesh(rows=2, cols=2)            # → meshRows=2, meshCols=2
grid = (mesh.rows, mesh.cols)             # → (2, 2)
matmul_simple[grid](A, B, C, ...)        # launches 2x2 = 4 tile programs
```

Each `(row, col)` in the grid is a physical AIE tile. Unlike GPU CuTile where the grid is dynamically scheduled on SMs, AIE programs have a fixed 1:1 mapping to physical tiles.

---

## CuTile vs Triton vs AIE C: Key Differences

| Aspect | CuTile | Triton | AIE C |
|--------|--------|--------|-------|
| **Data load** | `ct.load(A, index=(r,c), shape=(m,k))` — index-based | `tl.load(a_ptr + offsets)` — pointer arithmetic | `acquire_input_window(window_in_0)` — window API |
| **Data store** | `ct.store(C, index=(r,c), tile=val)` — index-based | `tl.store(c_ptr + offsets, val)` — pointer arithmetic | `acquire_output_window` + write + `release_output_window` |
| **Matmul** | `ct.mma(a, b, acc)` — explicit accumulator | `acc += tl.dot(a, b)` — returns value | `v4int8 MAC loop` |
| **Constants** | `ct.Constant[int]` — Python type annotation | `tl.constexpr` — custom annotation | C template params / `#define` |
| **K-loop count** | `ct.num_tiles(A, axis, shape)` — explicit | `range(0, K, BLOCK_K)` — implicit | `for(k=0; k<2; k++)` — hardcoded |
| **Accumulator init** | `ct.full(shape, val, dtype)` — generic | `tl.zeros(shape, dtype)` — zero-specific | BCF-assigned local buffer |
| **Type cast** | `ct.astype(tile, dtype)` — function call | `val.to(dtype)` — method call | In-kernel cast |
| **Block pointers** | Built-in: `index=` param on load/store | `tl.make_block_ptr` + `tl.advance` (Triton 2.0+) | Window pointers (compiler-managed) |
| **Swizzling** | `swizzle_2d()` — first-class API | Manual `pid_m`/`pid_n` decomposition | N/A (tiles are physically fixed) |

---

## Host Code Structure

The host section in `cutile_matmul.py` matches `proposal3.cc`'s `main()` line by line:

```
proposal3.cc                              cutile_matmul.py                          triton_matmul.py
────────────────────────────────────────  ────────────────────────────────────────  ────────────────────────────────────────
const int M=16, N=16, K=16;              M, N, K = 16, 16, 16                     M, N, K = 16, 16, 16
aieSetDevice(0);                         ct.set_device(0)                          aie_triton.set_device(0)
aieDim mesh(2, 2);                       mesh = ct.mesh(rows=2, cols=2)            mesh = aie_triton.mesh(rows=2, cols=2)
int32_t *A = malloc(M*K*sizeof);         A = np.arange(1, M*K+1, ...).reshape()   A = np.arange(1, M*K+1, ...).reshape()
for(i) A[i] = i+1;                      (done by np.arange)                       (done by np.arange)
int32_t *B = malloc(K*N*sizeof);         B = np.arange(1, K*N+1, ...).reshape()   B = np.arange(1, K*N+1, ...).reshape()
int32_t *C = malloc(M*N*sizeof);         C = np.zeros((M,N), dtype=np.int8)       C = np.zeros((M,N), dtype=np.int8)
matmul<<<mesh>>>(A,B,C,M,N,K);          matmul_simple[grid](A,B,C,tm=8,...)      matmul_simple[grid](A,B,C,M,N,K,...)
aieDeviceSynchronize();                  ct.synchronize()                          aie_triton.synchronize()
verify loop with printf                  numpy comparison + print                  numpy comparison + print
free(A); free(B); free(C);              (garbage collected)                        (garbage collected)
```

---

## AIE Constraints vs GPU CuTile

| Aspect | GPU CuTile | AIE CuTile |
|--------|-----------|------------|
| **Grid** | 1D/2D/3D, dynamically scheduled on SMs | 2D mesh, statically mapped to physical tiles |
| **Memory per program** | Shared memory: 48-228 KB | Tile-local data memory: ~32 KB |
| **Tile sizes** | Typically 64-256 | Typically 4-16 (int8) |
| **Data types** | fp16, fp32, bf16, int8 | int8, int16, int32 (AIEML native) |
| **Data movement** | Global → shared memory (software managed) | DDR → tile DMA (compiler managed, lock-synchronized) |
| **`ct.load` index** | Indexes into GPU global memory tiles | Indexes into DMA buffer descriptor configuration |
| **`ct.mma`** | Maps to tensor core WMMA/MMA instructions | Maps to AIE vector MAC (v4int8/v16int8) |
| **Swizzling** | Real performance benefit (L2 cache reuse) | No-op (tiles physically fixed) |
| **Ping-pong** | Software-managed shared memory staging | Hardware DMA ping-pong buffers with lock handshake |

---

## Design Decisions

1. **CuTile over Triton for AIE fit** — CuTile's index-based `ct.load(A, index=(row, col), shape=(m, k))` maps more naturally to AIE DMA buffer descriptor configuration than Triton's pointer arithmetic. The DMA hardware takes tile indices and shapes, not computed pointer offsets. This makes the CuTile→AIE mapping more transparent.

2. **Explicit accumulator in `ct.mma`** — CuTile's `acc = ct.mma(a, b, acc)` with an explicit accumulator parameter mirrors AIE's window-based acquire/compute/release pattern more closely than Triton's `acc += tl.dot(a, b)` which returns a value.

3. **`ct.num_tiles()` for K-loop** — CuTile makes the K-loop iteration count explicit from the tensor shape and tile dimensions, matching AIE's fixed iteration count in `for(k=0; k<2; k++)`. Triton computes this implicitly via `range(0, K, BLOCK_K)`.

4. **`ct.Constant[int]` for tile sizes** — More explicit than Triton's `tl.constexpr`, using standard Python type annotation syntax. Maps directly to compile-time constants needed for AIE DMA configuration.

5. **Two kernel variants** — `matmul_simple` gives the clearest 1:1 mapping to `proposal3.cc` for understanding the AIE programming model. `matmul_swizzle` demonstrates the full CuTile API including block swizzling (a no-op on AIE but standard on GPU). Both compile to identical routing IR.

6. **int8 data type** — Matches AIEML native vector width and `proposal3.cc`'s `input_window_int8` / `v4int8` types. The accumulator uses int32 to avoid overflow, with saturation to int8 on output.

7. **8x8 tile sizes** — Chosen to fit within 32 KB tile-local memory with ping-pong buffering. For the 16x16 matmul on a 2x2 mesh, this gives exactly 2 K-iterations (via `ct.num_tiles`), matching `proposal3.cc`'s `for (int k = 0; k < 2; k++)`.

8. **Host uses numpy** — numpy arrays replace C `malloc` + init loops. The `np.arange(1, M*K+1)` pattern exactly reproduces `for(i) A[i] = i+1`. Identical to the Triton version.

---

## `cuda.tile` Module API (adapted for AIE)

The `cuda.tile` module (imported as `ct`) exposes these functions:

| Function | Purpose | Triton Equivalent | Maps to (AIE C) |
|----------|---------|-------------------|------------------|
| `@ct.kernel` | Decorator marking an AIE tile kernel | `@aie_triton.jit` | `__global__` |
| `ct.set_device(id)` | Select AIE device | `aie_triton.set_device(id)` | `aieSetDevice(id)` |
| `ct.mesh(rows, cols)` | Define tile mesh dimensions | `aie_triton.mesh(rows, cols)` | `aieDim mesh(rows, cols)` |
| `ct.synchronize()` | Wait for all tiles to complete | `aie_triton.synchronize()` | `aieDeviceSynchronize()` |
| `ct.program_id(axis)` | Get tile row (0) or col (1) | `tl.program_id(axis)` | `get_coreid()` bit extraction |
| `ct.load(T, index, shape)` | Load tile from DDR via DMA | `tl.load(ptr + offsets)` | `acquire_input_window()` |
| `ct.store(T, index, tile)` | Store tile to DDR via DMA | `tl.store(ptr + offsets, val)` | `acquire_output_window()` + write + `release` |
| `ct.mma(a, b, acc)` | Matrix multiply-accumulate | `tl.dot(a, b)` + `+=` | `v4int8` / `v16int8` MAC loop |
| `ct.full(shape, val, dtype)` | Create constant-filled tile | `tl.zeros(shape, dtype)` | BCF-assigned tile-local address |
| `ct.num_tiles(T, axis, shape)` | Number of tiles along axis | `K // BLOCK_K` | Loop bound constant |
| `ct.astype(tile, dtype)` | Type cast (e.g., int32 → int8) | `val.to(dtype)` | In-kernel type conversion |
| `ct.Constant[int]` | Compile-time constant annotation | `tl.constexpr` | Template parameter |

---

## File Inventory

| File | Role |
|------|------|
| `cutile_matmul.py` | CuTile-style Python input (this design's reference implementation) |
| `triton_matmul.py` | Triton-style Python input (for comparison) |
| `proposal3.cc` | C equivalent using `__global__` + window API |
| `tilinglinalg_pipeline.h` | `TensorParam` struct + `buildRoutingIR()` / `runPipeline()` API |
