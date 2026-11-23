# Runtime Debug Level

(Moved from the root README.md.)

Control runtime diagnostic verbosity and feature flags from your source file using `#pragma aie_debug_level`:

```cpp
#pragma aie_debug_level 2

__global__ void mykernel(const int32_t *A, int32_t *B) {
    // kernel code
}

int main() {
    // host code
}
```

The debug level value is a bitfield:
- **Bits 0-3**: verbosity level (0-15)
- **Bits 4-31**: feature flags

## Verbosity Levels (bits 0-3)

| Level | Behavior |
|-------|----------|
| `0` | Silent (default when pragma is absent) |
| `1` | BD tracking and IO logs |
| `2` | Full diagnostics: DMA address log, write pattern, readback |

## Feature Flags (bits 4+)

| Flag | Bit | Value | Behavior |
|------|-----|-------|----------|
| `DISABLE_MULTID_DIM_DMA` | 4 | `16` | Suppress multi-dimensional DMA; force linear `__Runtime_dma_bd_config` |
| `DISABLE_PARTITIONTEARDOWN` | 5 | `32` | Skip `XAie_PartitionTeardown` in `__Runtime_device_teardown` |
| `MM2SBDFINISH_COUNTER` | 6 | `64` | Arm probe-tile MM2S ch0/ch1 BD-finished counters; read via `__Runtime_perfcnt_read_mm2s_probe`. Requires `--profiling` |
| `AIE_DMA_ISSUE_COUNT` | 7 | `128` | Use `XAie_PartitionInitialize_v2`/`_v2` teardown to arm DMA txn perf counters (whole partition, MM2S ch0) |
| `CORE_PERF_COUNTER` | 8 | `256` | Arm core-module cycle counters (active / vector-instr / stream-stall / lock-stall) on the probe tile. Requires `--profiling` |

## Examples

| Pragma Value | Verbosity | Flags | Effect |
|-------------|-----------|-------|--------|
| `0` | 0 | none | Silent, multi-dim DMA enabled |
| `1` | 1 | none | BD tracking, multi-dim DMA enabled |
| `2` | 2 | none | Full diagnostics, multi-dim DMA enabled |
| `16` | 0 | `DISABLE_MULTID_DIM_DMA` | Silent, multi-dim DMA suppressed |
| `18` | 2 | `DISABLE_MULTID_DIM_DMA` | Full diagnostics + multi-dim DMA suppressed |
| `128` | 0 | `AIE_DMA_ISSUE_COUNT` | Silent, DMA txn counters armed via `PartitionInitialize_v2` (MM2S ch0) |

The pragma is detected during preprocessing and emits a strong symbol override of `g_runtime_debug_level` into the generated `host.cc`. The runtime in `aie_runtime.c` defines this variable as a weak symbol with default `0`, so the linker picks the user-specified value when present.

This works for both single-tile (`aiehlc`) and multi-tile (`tilinglinalg`) compilation paths.
