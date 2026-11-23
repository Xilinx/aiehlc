<!-- Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
     SPDX-License-Identifier: Apache-2.0 -->
# Spatial Type Tags

Spatial type wrappers annotate `__global__` kernel parameters with their mesh
partitioning strategy. They control which mesh axis owns and replicates the
tensor — **not** which tensor dimension to split. `splitDim` always defaults
to 0; if a different tensor split dimension is needed, it should be specified
separately.

## Usage

```cpp
#include "aie_spatial_types.h"  // or use aiehlc built-in stubs

__global__ void matmul(
    aie::row_broadcast_in<input_window_int8*>  A,
    aie::col_broadcast_in<input_window_int8*>  B,
    aie::row_major_out<output_window_int8*>    C,
    int M, int N, int K)
{ /* compute */ }
```

Bare types (without `aie::` wrapper) default to `row_broadcast_in` (input) or
`row_major_out` (output) for backward compatibility.

## Input Distribution Types

| Spatial Type | Meaning | splitDim | hwAxisOwner | replicateOn |
|---|---|---|---|---|
| `aie::row_broadcast_in<T>` | Partition mesh by row, broadcast input (1-to-N) | 0 (default) | "row" | "col" |
| `aie::col_broadcast_in<T>` | Partition mesh by col, broadcast input (1-to-N) | 0 (default) | "col" | "row" |
| `aie::tiled_in<T>` | 1-to-1 unique tile mapping (no broadcast) | 0 (default) | "row" | "" |

## Output Recomposition Types

| Spatial Type | Meaning | splitDim | hwAxisOwner | replicateOn |
|---|---|---|---|---|
| `aie::row_major_out<T>` | Gather output row-major | 0 (default) | "row" | "col" |
| `aie::col_major_out<T>` | Gather output col-major | 0 (default) | "col" | "row" |
| `aie::row_reduce_out<T>` | Cascade reduction along row | 0 (default) | "row" | "" |

## Internal Representation

Spatial tags are parsed by the Clang frontend in `aiehlc.cc` and converted to
`TensorSplitDesc` / `SplitModel` structs that drive routing IR generation:

```cpp
struct TensorSplitDesc {
    int splitDim;            // tensor dimension to split (default: 0). Irrelevant when splitnum=1.
    std::string hwAxisOwner; // "row" | "col" | "" — which mesh axis owns this tensor's partition
    std::string replicateOn; // "row" | "col" | "" — which mesh axis to replicate/broadcast on
};

struct SplitModel {
    std::vector<TensorSplitDesc> tensorSplits;
    static SplitModel gemm();  // default GEMM split
    static TensorSplitDesc fromSpatialTag(const std::string &tag, bool isInput);
};
```

## Header File

`include/aie_spatial_types.h` defines the C++ struct templates. These are also
embedded as stubs in the aiehlc preprocessor for Clang AST parsing.

The kernel compiler (xchesscc) does not see spatial wrappers — they are stripped
from the kernel source text during `ExportFunction`.
