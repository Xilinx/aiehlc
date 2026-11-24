---
name: kerneltracehang
description: aiehlc / aiehlc-convtile "compile is stuck" right after `[KernelAnalysis]` with a /tmp/aiehlc_trace_*/h process at 100% CPU -- the kernel-analysis harness runs the kernel body on x86 and the body never ends (e.g. `for (int k = 0; i < 16; k++)`). Now bounded: `the kernel did not finish within N s ... last block operation it reached is at file:line`, AIEHLC_TRACE_TIMEOUT. Also: an app with `int main(int argc, char *argv[])` built lib<app>.a instead of the host ELF.
---

# Kernel analysis hangs on an endless loop in the kernel body

## Symptom

`source script/aiehlc.sh ... --runtime-source-file x.cc` prints the kernel
analysis start and then nothing, forever. `ps` shows

    sh -c -- ASAN_OPTIONS=... /tmp/aiehlc_trace_XXXX/h /tmp/aiehlc_trace_XXXX/trace1.txt 4 4 1 2
    /tmp/aiehlc_trace_XXXX/h ...        R+   (CPU time == wall time)

(Not the alib lock -- that one prints `waiting for alib lock`; skill
**alibflocklease**.)

## Cause

Kernel analysis (`runHarness`, `src/llvm/convautotile/kernel_trace.cpp`)
compiles the `__global__` body into an x86 harness (ASan) and runs it for every
core of the mesh, twice, to trace `load_buffer` / `store_buffer`. The user's
body really executes. A loop that never terminates -- the 2026-11-23 case was
`for (int k = 0; i < 16; k++) y[i] = ...` in `aiehlcpm_conv2d_2.cc` (tests `i`,
not `k`; `y[i]` stays in bounds, so ASan never fires) -- hung aiehlc, because
the run went through `std::system` with no limit.

## Fix

Each harness run is `timeout -k 5 <N>`, N = `AIEHLC_TRACE_TIMEOUT` (default
120 s, `0` = no limit). Exit 124/137 -> error:

    kernel analysis: the kernel did not finish within 120 s when traced on x86 (4x4 mesh,
    3 block operation(s) traced; the last block operation it reached is at x.cc:28 (core 0,0))
    -- most likely an endless loop in the kernel body ...

The harness installs a SIGTERM handler that flushes the trace file, so the
last traced `load_buffer` / `store_buffer` line (the loop is right after it)
survives the kill. Regression case: `kernel_trace_cases.py` "endless loop in
the kernel body (trace time limit)" (sets `AIEHLC_TRACE_TIMEOUT=3`).

A legitimately slow kernel on x86 (huge mesh, heavy scalar emulation): raise
`AIEHLC_TRACE_TIMEOUT`. Killing a stuck run by hand: kill the `/tmp/aiehlc_trace_*/h`
process, not aiehlc (`pkill -f` with that pattern also matches the shell that
runs it -- use `ps ... | awk '$5 ~ /aiehlc_trace_.*\/h$/'`).

## Same file, next trap: `int main(int argc, char *argv[])` -> lib<app>.a

`hostcompile.sh` chose ELF vs static library with `grep -q 'int main()'`, so a
main with parameters fell through to "library" and the build "succeeded" with
`Host static library built: .../lib<app>.a` and no `build/host`. Now
`grep -qE '^[[:space:]]*int[[:space:]]+main[[:space:]]*\('` (any parameters;
`void main()` and comments still don't match). Verified: `simplematmul2_aiegraph.cc`
(no main) still archives, `simpleconv2dkernel_port.cc` (`int main()`) and
`aiehlcpm_conv2d_2.cc` (`int main(int, char*[])`) link `build/host`.
