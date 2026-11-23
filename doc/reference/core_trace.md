# Core Trace

(Moved from the root README.md.)

Arm the AIE per-tile trace units from your source file using `#pragma aie_trace`.
Each pragma selects a tile `(col, row)` whose core-module and memory-module trace
units are set up in the generated `host.cc` (via `__Runtime_core_trace_begin_dma`).
The memory-module trace stream is drained through one of the tile's DMA channels;
the pragma lets you pick which one.

## Forms

```cpp
// 1. Default: trace tile (0,3); mem-trace drains via S2MM channel 0.
#pragma aie_trace(0, 3)

// 2. Ranges: trace every tile in col 1..2, row 3 (all get the default S2MM ch0).
#pragma aie_trace(1:2, 3)

// 3. STREAM: pick the mem-trace DMA direction + channel explicitly.
#pragma aie_trace((0, 3), (STREAM, "s2mm", 0))
#pragma aie_trace((0, 3), (STREAM, "mm2s", 1))

// 4. PARAMETER: name a kernel window; the compiler resolves the physical
//    (direction, channel) the tiling flow assigned to it on that tile.
//    Input windows drain via S2MM, output windows via MM2S.
#pragma aie_trace((0, 3), (PARAMETER, "win_a"))
```

## Selection semantics

| Form | `mem_dma_kind` | `mem_dma_ch` |
|------|----------------|--------------|
| `aie_trace(col, row)` | `S2MM` | `0` (default) |
| `(STREAM, "s2mm", ch)` | `S2MM` | `ch` |
| `(STREAM, "mm2s", ch)` | `MM2S` | `ch` |
| `(PARAMETER, "win_x")` | resolved: input→`S2MM`, output→`MM2S` | resolved channel |

- The direction string is case-insensitive (`"s2mm"` / `"S2MM"`).
- `PARAMETER` resolution reads the `dfschedule.config.create_io` provenance to
  map a window name to the DMA channel the flow physically assigned on the traced
  tile. It is available in the multi-tile (`tilinglinalg`) flow; if a name cannot
  be resolved (unknown window, or no matching create_io on that tile) the compiler
  warns and falls back to the default `S2MM ch0` so the trace still arms.
- Pragmas are repeatable and ranges expand to one trace-setup call per tile; the
  selected mem-DMA applies to every tile expanded from that pragma.
- The legacy flat form `#pragma aie_trace(0, 3)` (no inner parens) is still
  accepted and is identical to the default `S2MM ch0` behaviour.

Each pragma emits `__Runtime_core_trace_begin_dma(dev, col, row, dma_kind, dma_ch)`
into `host.cc`. STREAM selection works in both single-tile and multi-tile flows;
PARAMETER resolution requires the multi-tile provenance and otherwise falls back
to the default.

## Validation

The mem-DMA selection is checked at compile time and the build **fails with an
error** (non-zero exit) rather than silently mis-tracing when:

| Case | Example | Error |
|------|---------|-------|
| Invalid STREAM direction | `(STREAM, "s3mm", 0)` | `STREAM direction "s3mm" is invalid (expected "s2mm" or "mm2s")` |
| STREAM channel not used by the app on that tile | `(STREAM, "mm2s", 1)` when the tile only drives `mm2s ch0` | `STREAM mm2s ch1 is not used by the app on tile(0,3). Available mm2s channels: ch0` |
| PARAMETER name is not a kernel window | `(PARAMETER, "win_xyz")` | `PARAMETER "win_xyz" is not a kernel window/port name.` |

The STREAM channel check compares the requested `(direction, channel)` against the
channels the tiling flow actually assigned on the traced tile (from the
`dfschedule.config.create_io` provenance), so it rejects both hardware-invalid
indices and valid-but-unused channels. Tiles a given kernel does not own are
skipped so multi-kernel meshes are not spuriously failed. The `Default` form
(`aie_trace(col, row)`, S2MM ch0) is not validated.
