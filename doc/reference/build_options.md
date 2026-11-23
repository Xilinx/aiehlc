# Build Options

(Moved from the root README.md.)

Two host-build toggles, both off by default and both compile-time (they change how
`aie_runtime.c` is compiled, so they need a rebuild to take effect):

| Flag | Define | Effect |
|------|--------|--------|
| `--profiling` | `-DAIEHLC_PROFILING=1` | Compile in the PS PMU profiling layer: launch-phase timers (kload / bdcfg / coreen / startio / wait_io) and the `[PERF]` summary. Off, the `RT_PROF_*` macros expand to nothing and the reader functions return zeros. |
| `--skip-bss` | `-DAIEHLC_SKIP_BSS_DEFAULT=1` | Load only PT_LOAD segments with file bytes, skipping pure-`.bss` zero-init on kernel load. Safe only when no global is read before being written — the DMA-fed window buffers qualify. |

```bash
source script/aiehlc.sh --platform baremetal --aie-version 5 --profiling --skip-bss \
    --runtime-source-file ./example/tileprogram/ccode/simplematmul2_prof.cc
```

Note the `MM2SBDFINISH_COUNTER` and `CORE_PERF_COUNTER` pragma bits above are inert
without `--profiling`: the counters are armed by code that only exists in a profiling
build, so the pragma alone reports zeros.

`--skip-bss` must be the compile define rather than the `AIEHLC_SKIP_BSS` environment
variable on baremetal, where newlib `getenv()` always returns `NULL`.
