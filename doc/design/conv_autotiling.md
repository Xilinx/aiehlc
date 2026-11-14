# Conv2d auto-tiling: declarative `aie::Conv2d` + `aie::Mapping`

Status: implemented — `src/llvm/convautotile/` (model + desugaring, linked into
`aiehlc`), `aiehlc-convtile` (CLI), `src/frontend/convtune/` (AutoTVM/XGBoost tuner).
Example: `example/tileprogram/ccode/simpleconv2d_new_load.cc`.

## 1. What the user writes

```cpp
constexpr aie::Conv2d conv = {.h=224, .w=224, .c=3, .k=64, .kh=7, .kw=7, .stride=2, .pad=3};
constexpr aie::Mapping map = {
  .space = { aie::split(aie::OH, MESH_ROW),     // mesh rows
             aie::split(aie::OC, MESH_COL) },   // mesh cols
  .time  = { aie::tile(aie::OH, 7),             // optional: omit -> autotune
             aie::tile(aie::OW, 28) },
  .hints = {.pad = aie::PadMaterialize::DDR, .im2col = aie::Im2col::Dma,
            .pp_depth = 2, .l1_budget = aie::Bytes{4096}}};

__global__(conv, map) void conv2d_spatial(
    aie::port<input_window_int8 *, aie::IFM> win_a,
    aie::port<input_window_int8 *, aie::WEIGHTS> win_b,
    aie::port<output_window_int8 *, aie::OFM> win_c) { ... uses SP_KH, TILE_W, OW_T ... }
```

instead of hand-writing three `aie::GemmSpace`s (d1/d2/d3/d4, halo slice/step,
`slice_tiling`), a `GlobalPolicy`, and ~25 mirrored `#define`s
(`simpleconv2d.cc`).

* `aie::Conv2d` fields: `h w c k kh kw stride pad`, optional `dilation` (1) and
  `c_align` (default: `c` rounded up to 4, the int8 channel-layout stride).
* Split/tile factors are integer constant expressions (object-like `#define`s
  from the file and the headers it includes from its own directory --
  `#include "x.h"` or `#include <x.h>` -- are resolved), or `MESH_ROW` / `MESH_COL`,
  which resolve to the launch mesh of `device.partition({...}, ROWS, COLS)`.
  An explicit factor that disagrees with that mesh is an error.
* Supported mapping (what the tilinglinalg spatial-halo path implements):
  `space = {split(OH, R), split(OC, C)}`, `time ⊆ {tile(OH, th), tile(OW, tw)}`,
  `pad = DDR`. Anything else is rejected with a message naming the field.

## 2. Where the logic lives

```
user .cc ──inlineSourceIncludes──► applyConvAutoTiling (aiehlc.cc)
                                     └─ convautotile::desugarConvMapping   (conv_desugar.cpp)
                                          ├─ parse designated initializers + macros + mesh
                                          ├─ convautotile::derive            (conv_autotile.cpp)
                                          └─ rewrite text:
                                               #define SP_KH 7 ... TILE_W 61 ...
                                               constexpr aie::GemmSpace __aie_<k>_{IFM,WEIGHTS,OFM}
                                               constexpr aie::GlobalPolicy __aie_<k>_policy
                                               __global__(__aie_<k>_policy) ... port<T, __aie_<k>_IFM>
         ──► existing #define scan, __global__ rewrite, Clang AST GemmSpace path (unchanged)
```

The pass runs on the text **before** every other aiehlc rewrite, so the rest of
the compiler sees exactly the hand-written form (`aout/newfile.cpp` shows it;
the original declarations are kept as comments). The generated `#define`s reach
the kernel translation unit through the normal user-macro path. A `#define` the
user already wrote is not re-emitted (a differing value is warned about).

The model is dependency-free C++17 so the same code backs `aiehlc`, the
`aiehlc-convtile` CLI and the unit test.

## 3. The model

Every halo quantity is the footprint of an output tile under `ih = oh·S + kh·D`:

```
slice(T) = (T − 1)·S + (K − 1)·D + 1        step(T) = T·S        overlap = slice − step
```

applied twice — mesh split (per-core rows `OH/R`) and on-core time tile:

| Port | dim | fullsize | tile_round | tile_size | stride | slice_tiling |
|------|-----|----------|------------|-----------|--------|--------------|
| IFM | d1 (H) | H+2P | R | slice(OH/R) | (OH/R)·S | slice(th) / th·S / (OH/R)/th |
| IFM | d2 (W) | W+2P | OW/tw | slice(tw) | tw·S | — |
| IFM | d3 (C) | c_align | — | c_align | — | padsize = c_align − c |
| WEIGHTS | d1 (OC) | K | C | K/C | K/C | — |
| WEIGHTS | d2/d3/d4 | KW / KH / c_align | | | | |
| OFM | d1 (OH) | OH | R | OH/R | OH/R | th / th / (OH/R)/th |
| OFM | d2 (OW) | OW | OW/tw | tw | tw | — |
| OFM | d3 (OC) | K | C | K/C | — | — |

For the stem this reproduces `simpleconv2d.cc` field for field (61/56, 19/14×4,
61/56×4, …) — the unit test pins every value, and the generated `routing.cc`,
`kernel.cc`, `aieml.bcf`, `aie_resource_map.h` and the whole generated schedule
part of `host.cc` are **byte-identical** to the hand-written file's build.

### Legality (each rule names its source)

| Rule | Why |
|------|-----|
| `OH % R == 0`, `K % C == 0`, `(OH/R) % th == 0`, `OW % tw == 0` | the halo path has no tails |
| reads stay in the padded input; on-core rounds tile the mesh slice exactly | halo bounds |
| `pp_depth·(ifm + wgt + ofm) ≤ 49152` | pipeline "Buffer memory check … / 49152" |
| `ifm + wgt + ofm + 320 ≤ 32768 − 1024` | reference kernels copy each window to the stack (`_stack DM_stack 0x8000`; 11200 B measured in `kernel.calltree` for 7×28); skill **aiecorestackoverflow** |
| `tw < OW` (≥ 2 width rounds) | `w_rounds = 1` takes the legacy path → `conv spatial-halo M-tile mismatch` |
| OFM tile and WEIGHTS ≤ `l1_budget` | non-halo ports are clamped to `l1_budget` (`maxPingPongBytes`) and chunked into `ceil(bytes/l1_budget)` rounds of `l1_budget` — uneven, overruns. The halo IFM is exempt. |
| `pp_depth · l1_budget · 3 ≤ 49152` | the pipeline's own check (`Memory budget exceeded! ... maxPingPongBytes=… numTensors=3`) sizes every tensor's buffers at `l1_budget` |
| `2·(ifm + wgt + ofm) + 3072 ≤ 32768` | link: `CoreMemAllocator(0x78000, 0x8000)` -- the port buffers get a 32 KB region (the other 32 KB is `DM_stack`), the last 2 KB of it is the `klog` ring, statics share it; `kernel.cc` declares ping **and** pong for every window at any `pp_depth`. Past it: `could not find free space for SpaceSymbol buf_in_ping_0 in memory DMb` (skill **dmbufferregion**) |

The `tw < OW` and `l1_budget` rules were found by building derived tilings, the
last two by building `offload_prototype` backbones for ResNet-18 layer 4
(64 -> 64, 3x3); the tuner re-checks them on every build (§5). Kernel-only
sources (§5 of op_mapping_programming_model.md) get `l1_budget` sized to the
kernel's own weight / output blocks (>= 4 KB) and `pp_depth = 1` when double
buffering cannot pass the pipeline check.

### Cost model

`est = max(core, dma) + rounds·600` with
`core = MACs / (21/68) + copied_bytes·7` (scalar reference kernel,
`doc/performance/conv2dstem_perf_review.md`), `dma = bytes / 2.3`,
at 998 MHz. It is only used to rank candidates, to pick `.time` when the user
omits it, and as the tuner's offline "analytic" cost. With the scalar kernel
every legal tiling is within ~0.5 % (compute-bound); the halo shape starts to
matter once the kernel is vectorized — which is why the tuner's **board** mode
measures instead of trusting the model.

## 4. Knobs (`AIEHLC_CONV_TUNE`)

`AIEHLC_CONV_TUNE='{"time_oh":14,"time_ow":16}'` (or `=@file.json`) overrides the
mapping before derivation. Keys: `time_oh`, `time_ow`, `pp_depth`, `l1_budget`,
`c_align`. aiehlc prints `[ConvAutoTile] <kernel>: … [AIEHLC_CONV_TUNE]` and writes
`aout/conv_autotile.json` (derived ports, geometry, features, kernel defines).
A rejected configuration aborts aiehlc with the rule that failed.

## 5. Tuning with TVM AutoTVM / XGBoost (`src/frontend/convtune`)

`create_task(src)` builds an AutoTVM `Task` whose `ConfigSpace` has the knobs
`time_oh`, `time_ow`, `pp_depth`, `l1_budget`, restricted by `multi_filter` to
the model-legal set — every d1/d2/d3 is derived from the knobs, so no tuner can
emit an inconsistent halo. `XGBTuner(feature_type="knob")` learns on the tile
sizes directly (also `xgb_rank`, `random`, `grid`, `ga`). Measurement is a
custom Builder/Runner:

| mode | build | cost |
|------|-------|------|
| `analytic` | none | model estimate |
| `build` | `aiehlc.sh` with `AIEHLC_CONV_TUNE` per trial, + pipeline-log cross-check | model estimate |
| `board` | as `build` | `run_cmd` (default `apppaltest.py {elf} -y --netload --netload-no-force`) → `[BENCH] aie_wall_ms`; a run without `PASS: all … match` or with `FAIL/MISMATCH/TIMEOUT` is an error, never a fast result |

The cross-check (`check_pipeline_log`) requires every port to be one round of
exactly the modelled size and the pipeline's buffer total to equal the model's
L1; drift between model and pipeline fails the trial loudly.

Builds/runs are sequential (`aiehlc.sh` wipes `thirdparty/alib/lib` per run).

```bash
PYTHONPATH=src python -m frontend.convtune space  example/tileprogram/ccode/simpleconv2d_new_load.cc
PYTHONPATH=src python -m frontend.convtune tune   example/tileprogram/ccode/simpleconv2d_new_load.cc \
        --mode board --n-trial 16 --l1-budgets 4096,8192 --workdir /tmp/ct
AIEHLC_CONV_TUNE=@/tmp/ct/best_knobs.json source script/aiehlc.sh --aie-version 5 --runtime-source-file ...
```

Outputs: `best.json` (best knobs, cost, full derived ports, per-trial history),
`best_knobs.json` (feed to `AIEHLC_CONV_TUNE`), `tune.log` (AutoTVM records,
usable with `autotvm.record` tooling), `trials/NNNN_*/` (build and run logs).

## 6. Tests

* `src/llvm/convautotile/unitest/run.sh` — golden values of `simpleconv2d.cc`,
  footprint/dilation rule, every rejection rule, autotune, overrides, desugaring
  of the real example, `MESH_ROW`/`MESH_COL`.
* `PYTHONPATH=src python src/frontend/convtune/test_convtune.py` — CLI wrapper,
  ConfigSpace == legal set, XGBTuner reaches the model optimum (analytic mode).

## 7. Limits / next steps

* Mapping coverage is what the spatial-halo path implements (OH × OC mesh, OH/OW
  time tiles). New downstream capabilities (tails, IC split with accumulation,
  sliding-window reuse) should land as new legality cases in `deriveFixed` plus a
  knob, not as hand-written d-dims.
* The stack rule models kernels that copy their windows; a zero-copy kernel can
  pass `--stack-budget 0 --copy-cycles-per-byte 0` to the tuner/CLI.
