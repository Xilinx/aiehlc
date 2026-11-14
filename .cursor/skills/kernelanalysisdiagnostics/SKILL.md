---
name: kernelanalysisdiagnostics
description: Misspelled/missing `#include` reported as `no #define HW_ROWS` instead of `file not found` (source-text passes run before Clang). Kernel-only block-load kernel (`aie::port<..., aie::shape<...>>`) fails with `[KernelAnalysis] ERROR: kernel analysis needs the launch mesh: no device.partition({...}, ROWS, COLS) found` although the source DOES call `device.partition(...)`; mesh ROWS/COLS are macros from `#include <x.h>`; or xchesscc says `no matching function for call to 'aiehlc_load_buffer'` / `aiehlc_release` / `aiehlc_commit` / `candidate function not viable: no known conversion from 'input_window_int8 *' to 'input_window_int8'`. Also how to add a source-level diagnostic so a mistake is reported where aiehlc can see it, not later in xchesscc.
---

# Kernel analysis: unresolved mesh, by-value windows

Seen with `example/other/aiehlcpm_gemm1.cc` (2026-11-12). It failed with one
misleading line, and each fix then exposed the next mistake.

## 1. "no device.partition found" when there is one
The mesh is read from the **source text** (`findMeshShape`,
`src/llvm/convautotile/conv_desugar.cpp`), not from clang. Macros come from
`collectMacros`, which used to follow **only `#include "x.h"`**. With
`#include <aiehlcpm_gemm1.h>`, `HW_ROWS` / `HW_COLS` were unknown, the
partition arguments did not evaluate, and the error claimed the call did not
exist.

Fixed: `collectMacros` follows `<x.h>` as well when the file exists in the
source's directory (otherwise it is a system header and is skipped). This
matches the build, because `aiehlc.sh` adds that directory with `-I`.
`MeshShape::why` / `sourceLaunchMesh(src, dir, &why)` now give the cause:

    kernel analysis needs the launch mesh: device.partition({3, 6, 0, 6}, MY_ROWS, 4):
    ROWS `MY_ROWS` is not an integer constant expression -- no `#define MY_ROWS` in the
    source or in a header #included from the source's directory ...

Still not followed: headers from another directory and nested `-I` paths.
For those, use literals or define the macro in the `.cc`.

## 1b. Misspelled / missing header reported as an undefined macro
`#include <aiehlcpm_gemm1——3.h>` (no such file) used to report
``ROWS `HW_ROWS` ... no `#define HW_ROWS` `` and never Clang's "file not
found". The reason is the order of operations: `applyConvAutoTiling`, where
KernelAnalysis runs, works on the source **text** before Clang preprocesses
it. `collectMacros` skips any header it cannot open, and KernelAnalysis then
calls `exit(1)`, so Clang never runs.

Fixed: `MyFrontendAction::checkSourceIncludes` (`src/llvm/aiehlc.cc`) runs
before `inlineSourceIncludes` / `applyConvAutoTiling`. It resolves every
`#include` outside `#if`/`#ifdef` against the source dir, Clang's
`HeaderSearchOptions::UserEntries` (the -I/-isystem paths) and the resource
dir, then prints Clang's own message and stops:

    .../aiehlcpm_gemm1.cc:1:10: fatal error: 'aiehlcpm_gemm1——3.h' file not found
    [aiehlc] ERROR: 1 #include(s) not found in .../example/other or the 21 include path(s) Clang searches ...

Conditional includes are not checked, because they may belong to another
platform; Clang reports those itself. Checked for false positives on
`aiehlcpm_gemm1.cc`, `simplegemm_kernel_port.cc`, `simplematmul.cc` and
`simpleconv2d_new_load.cc` (all rc=0).
General rule: a source-text pass that runs before Clang must not turn
something Clang would diagnose into a different error of its own.

## 2. By-value window
`aie::port<input_window_int8, aie::IFM, ...>` (no `*`) passes the frontend and
the x86 trace. It then fails in xchesscc with `no matching function for call to
'aiehlc_load_buffer'`. `parsePort` (`kernel_trace.cpp`) now rejects it:
`port a: the window type must be a pointer -- write aie::port<input_window_int8 *, ...> a`.

## Other mistakes in that file (all reported clearly)
- An undeclared variable (`w`): `the kernel body does not compile for tracing ... 'w' was not declared`.
- An OFM declared `input_window_int8`: `aie::store_buffer on an input port`.
- A typo in a host function name: a clang error that points at `newfile.cpp:<line>`, which is the
  rewritten copy in `aout/`, not the user's file.

## Adding a diagnostic
Report the mistake at the first stage that can see it: text parse
(`parsePort` / `findMeshShape`), then trace, then plan. An error from xchesscc
or the board means a check is missing earlier. Add a case to
`src/llvm/convautotile/unitest/kernel_trace_cases.py` (`GEMM_CASES`) that
expects the new text, then run it with
`python3 .../kernel_trace_cases.py build/aiehlc-convtile gemm:`.
