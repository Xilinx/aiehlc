<!-- Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
     SPDX-License-Identifier: Apache-2.0 -->
# rcom — ROCm/HIP GEMM → AIE (example directory)

This directory is the **ROCm/HIP front-end example** for AIEHLC. It holds the
canonical HIP GEMM input and the AIE source that `rcom` generates from it:

```
example/tileprogram/rocm/
├── matmul_hip.cpp     # input:  canonical ROCm/HIP int8 GEMM (the "user" source)
├── gen/               # output: AIE source emitted by rcom (default --out here)
│   ├── matmul.cc      #   kernel (cache-A / stream-B template) + host main()
│   └── matmul.h       #   M/K/N + HW mesh dims, scalar_matmul + verify_matmul
└── README.md          # this file
```

`rcom` turns the HIP source into a CUDA-style `.cc` + `.h` (the same dialect as
`example/tileprogram/ccode/simplematmul2.cc`) and then drives the **unchanged**
`script/aiehlc.sh` pipeline to build host + kernel ELFs:

```
matmul_hip.cpp ──(rcom.py)──► gen/matmul.cc + gen/matmul.h ──(aiehlc.sh)──► aout/main.elf
```

It is a **recognize-and-template** front end, not a source-to-source
translator: rcom confirms the HIP kernel is a GEMM, then emits the proven AIE
matmul template rather than translating the thread-indexed HIP body
line-for-line.

> The tool itself (`src/mlir/mlirfront/frontend/rcom/rcom.py`) has a fuller
> reference in
> [`src/mlir/mlirfront/frontend/rcom/README.md`](../../../src/mlir/mlirfront/frontend/rcom/README.md).
> This file documents the example directory and the end-to-end flow.

---

## How to use

Initialise the toolchain first:

```bash
source script/setup.sh
```

### One-shot: HIP source → ELF

```bash
source script/rcom.sh --rocm-source-file example/tileprogram/rocm/matmul_hip.cpp \
    --aie-version 5
# → generates example/tileprogram/rocm/gen/matmul.{cc,h}, then builds aout/main.elf
```

`rcom.sh` options (mirror `aiehlc.sh` arg style):

| Option | Default | Meaning |
|---|---|---|
| `--rocm-source-file <hip>` | (required) | Path to the HIP GEMM source. |
| `--aie-version V` | `5` | Passed through to `aiehlc.sh` (`5` = AIE2PS). |
| `--mesh RxC` | `4x4` | AIE tile mesh rows × cols. |
| `--name NAME` | `matmul` | Generated kernel / function name. |
| `--out DIR` | `example/tileprogram/rocm/gen` | Output dir for `<name>.cc/.h`. |

### Two-step: generate, inspect, then build

```bash
# 1. Emit AIE source only (no build)
python3 src/mlir/mlirfront/frontend/rcom/rcom.py example/tileprogram/rocm/matmul_hip.cpp \
    --emit-only --out example/tileprogram/rocm/gen
#    → gen/matmul.cc  and  gen/matmul.h

# 2. Build with the normal pipeline
source script/aiehlc.sh --aie-version 5 \
    --runtime-source-file example/tileprogram/rocm/gen/matmul.cc
```

### Run on hardware (data check)

```bash
python3 ./script/test/apppaltest.py -y -nonreboot > ./applog 2>&1
# PASS: "PASS: all ... match" + "device_teardown done", no "AIE ERROR"
```

> **Why `gen/` lives here, not in `aout/`:** `aiehlc.sh` runs
> `rm -rf $(pwd)/aout/` at startup, so the generated files must sit **outside**
> `aout/`. `rcom.sh` defaults `--out` to this directory's `gen/`. If you pass
> `--out` yourself, keep it outside `aout/`.

---

## Architecture

### End-to-end flow

```
 ┌───────────────────┐   ┌──────────────────────┐   ┌──────────────────┐
 │  matmul_hip.cpp   │   │      rcom.py         │   │   aiehlc.sh      │
 │  (ROCm/HIP GEMM)  │──►│  parse → tile → emit │──►│  Clang→MLIR→     │──► aout/main.elf
 │  int8, M=N=K=256  │   │  gen/matmul.{cc,h}   │   │  EmitC → ELFs    │   (host + kernel)
 └───────────────────┘   └──────────────────────┘   └──────────────────┘
```

`rcom.py` is a dependency-free (`stdlib`-only) tool with three stages plus an
optional build delegation:

1. **`parse_hip(src)`** — strips comments, finds `__global__ void NAME(...)`,
   classifies pointer params (int8 only in v1), decides direction (the pointer
   written via `NAME[...] = ...` is the output C; the other two are inputs A, B),
   reads dims M/N/K from `#define`/`const int`, and validates the body is a real
   GEMM (output store + MAC `sum += a*b` + a `k` loop). Non-GEMM input fails
   fast with a `ParseError` so there is no silent wrong output.
2. **`compute_tiling(M, N, K)`** — returns `(m_tile, n_tile, k_chunk)`. For the
   HW-validated `M=N=K=256` it reproduces the proven `16/16/64`. Other dims use
   heuristics (`tile≈16`, `K%chunk==0`) and print a warning (untested on HW).
3. **`emit_cc` / `emit_h`** — `@@TOKEN@@` substitution into templates derived
   verbatim from `simplematmul2.cc` + `simplematmul.h`.
4. **build delegation** (`--build` / `rcom.sh`) — shells out to
   `script/aiehlc.sh`. rcom makes **no** C++/MLIR changes; it sits strictly in
   front of the existing pipeline.

### What the generated files contain

**`gen/matmul.h`** — GEMM dims (`#define M/K/N`), HW mesh dims
(`#define HW_ROWS/HW_COLS`), a pure-scalar `scalar_matmul` reference, and
`verify_matmul` (compares AIE output C against the CPU reference, prints
A/B/C/C_ref and PASS/FAIL).

**`gen/matmul.cc`** — the AIE kernel + host:
- Three composition-based `aie::GemmSpace` policies, one per port:
  - `RowBA` — A `[M,K]`, broadcast activations, row layout (`d1`=M-tile,
    `d2`=K-chunk).
  - `ColBB` — B `[N,K]`, broadcast weights, col layout (`d1`=N-tile,
    `d2`=K-chunk).
  - `LtoR_Merge` — C `[M,N]`, left-to-right merge (`d1`=M-tile, `d2`=N-tile).
- The `__global__(matmul_policy)` kernel: cache-A / stream-B matmul —
  M-sub-tile × K-round loops accumulate `int16`, saturate to `int8`, emit C.
- Host `main()`: `aieSetDevice`, `device.partition({0, cols-1, 0, 6}, HW_ROWS,
  HW_COLS)`, `device.alloc` for A/B/C, init loops, `matmul<<<mesh>>>`, then
  `verify_matmul`.

### HIP → AIE `.cc` mapping

| HIP construct | Generated AIE `.cc` |
|---|---|
| `__global__ void gemm(A,B,C,M,N,K)` | `__global__(matmul_policy) void matmul(port<...,RowBA> win_a, ...ColBB win_b, ...LtoR_Merge win_c)` |
| int8 pointer params | `input_window_int8*` / `output_window_int8*` |
| thread-indexed `sum+=A*B; C=sat(sum)` | proven cache-A / stream-B kernel body |
| host dims `M,N,K` | `#define M/K/N` in `matmul.h` |
| launch grid/block | `#define HW_ROWS/HW_COLS` + `device.partition({0,cols-1,0,6}, HW_ROWS, HW_COLS)` |
| `hipMalloc`/`hipMemcpy`/launch/verify | host `main()`: `device.alloc`, init loops, `matmul<<<mesh>>>`, `verify_matmul` |

---

## Data semantics

`C[M×N] = A[M×K] * B^T[N×K]`, int8, saturating to int8. **B is stored
row-major as `B[N][K]`** (already transposed relative to textbook `C = A*B`),
matching the AIE `simplematmul2.cc` layout. The sample input fills
`A[i] = (i%7)-3`, `B[i] = (i%5)-2` for a deterministic verify.

## Limitations (v1)

- **int8 only** — other dtypes error out in the parser.
- Only the **`M=N=K=256`, 4×4 mesh** config is HW-validated; other dims/mesh use
  heuristic tiling and print a warning (untested on hardware).
- The kernel body is **templated, not faithfully translated** — a custom HIP
  GEMM epilogue is recognized as a GEMM but emitted as the standard template.

## Related files

| Path | Role |
|---|---|
| `matmul_hip.cpp` | Sample canonical HIP int8 GEMM input. |
| `gen/matmul.cc`, `gen/matmul.h` | Generated AIE source (do not edit; regenerate). |
| `src/mlir/mlirfront/frontend/rcom/rcom.py` | The tool (parser + tiling + generator + CLI). |
| `src/mlir/mlirfront/frontend/rcom/README.md` | Full tool reference. |
| `script/rcom.sh` | Wrapper: `rcom.py --emit-only` then `source aiehlc.sh`. |
| `example/tileprogram/ccode/simplematmul2.cc` | The proven template the output mirrors. |
