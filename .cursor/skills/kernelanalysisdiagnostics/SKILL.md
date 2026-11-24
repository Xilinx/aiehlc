---
name: kernelanalysisdiagnostics
description: Misspelled/missing `#include` reported as `no #define HW_ROWS` instead of `file not found` (source-text passes run before Clang). Kernel-only block-load kernel (`aie::port<..., aie::shape<...>>`) fails with `[KernelAnalysis] ERROR: kernel analysis needs the launch mesh: no device.partition({...}, ROWS, COLS) found` although the source DOES call `device.partition(...)`; mesh ROWS/COLS are macros from `#include <x.h>`; or xchesscc says `no matching function for call to 'aiehlc_load_buffer'` / `aiehlc_release` / `aiehlc_commit` / `candidate function not viable: no known conversion from 'input_window_int8 *' to 'input_window_int8'`. Per-core `offset has 4 and shape 4 dims, but the tensor has 3` / `block dim 2 covers [16, 32) outside the tensor extent 4` on every mesh column when the port's `aie::shape` (or role) is wrong -- now a `hint:` line naming the declaration. Per-core `offset has 4 and shape 4 dims, but the tensor has 3` / `block dim 2 covers [16, 32) outside the tensor extent 4` on every mesh column when a port's `aie::shape` (or role) is wrong -- now a `hint:` line naming the declaration. Also how to add a source-level diagnostic so a mistake is reported where aiehlc can see it, not later in xchesscc.
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

Two declaration forms give the mesh (2026-11-23): `device.partition({...}, R, C)`
/ `partition(R, C)`, and `aieArray device(R, C);` launched as `k<<<device>>>` (or
`aieMesh mesh = device;`). Before, the constructor form failed with "no
device.partition(...) call was found". The text side is `findMeshShape`
(partition call first, then an `aieArray` / `aieDim` `name(R, C)` declaration;
`aieArray device;` without arguments has no shape); the AST side
(`aiehlc.cc`, `__aie_launch` visitor) follows the launch argument through the
implicit `aieArray -> aieMesh` conversion (`peelMeshExpr`) to `device`'s
constructor. The stubs (`aiehlc.cc` x2, `tilinglinalg_pipeline.cpp`,
`aiegraph/kemu.py`) give `aieArray` the `(rows, cols)` constructor + `operator
aieMesh()`; aiekernel lint L015 and `aiegraph/aiperf.py` accept the form too.
Cases: `test_trace_routing.cpp` `testLaunchMeshForms`.

Also fixed then: the AST read **no** dims from the 2-argument
`device.partition(R, C)` (it only matched 3 arguments), so the pipeline fell back
to a **2x2** mesh (`MeshKernelDesc ... mesh=0x0`, `routingcreatehwmesh row = 2,
col = 2`) while kernel analysis traced R x C. All three forms now give the same
kernel.cc / routing.cc / bcf / prx (verified on `simpleconv2dkernel_port.cc`).
`example/tileprogram/ccode/simple.cc` (`partition(4, 4)`, kernel sized for 4x4)
now builds 4x4 instead of 2x2 -- not yet re-run on a board.

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

## 3. A wrong port declaration reported only as per-core symptoms

Seen with `example/other/aiehlcpm_conv2d.cc` (2026-11-16). The weights port
was declared with the input's shape, `aie::IFM, aie::shape<230,230,4>> b`, and
the output with 4 channels instead of 64. The harness (`block()` in the
trace runtime) checks each `load_buffer` / `store_buffer` call against the
declared shape, so the only output was one line per core and symptom:
`offset has 4 and shape 4 dims, but the tensor has 3`, and four
`covers [16, 32) outside the tensor extent 4` lines, one per mesh column.
Nothing pointed at the declaration.

Fixed: `declaredShapeHints()` (`kernel_trace.cpp`) runs before the per-core
errors and adds at most one line per port:

    b: hint: the kernel's blocks are 4-D, spanning (64,7,7,4), which no offset fits into the
       declared `aie::shape<230,230,4>` (line 4) -- check this port's aie::shape
       (the blocks suggest aie::shape<64,7,7,4>) and its role (IFM)
    a: hint: its load/store passes an offset of 4 and a block shape of 3 coordinates -- the
       call has an extra offset coordinate (the port declares `aie::shape<230,230,4>` (line 3), 3-D)

When the hint fires (deliberately narrow):
- **declaration**: the offset and shape agree in rank, and either that rank
  differs from the declaration or some block is larger than a declared dim.
  Then no offset can fit, so the declaration is the likely mistake. The
  suggestion is the hull `max(off + shape)` over all blocks, omitted when an
  offset is negative (a conv border; that is the `.pad` hint instead).
- **call**: the offset and shape differ in rank, so the call itself is
  inconsistent.
- **no hint**: the block fits the declaration but an offset runs past the end
  (e.g. `pid_c * oc_t * 20`). The call is as likely wrong as the declaration,
  so only the per-block error is printed.

Regression: three `kernel_trace_cases.py` cases whose names contain `hint:`.

## 4. Errors that named the port but not the line to fix (2026-11-17)

Same file, after section 3. The `a` hint said "the call has an extra offset
coordinate" but pointed only at the port's **declaration** (line 3). The broken
call was on line 19. The never-released error named no line at all. Cause: the
trace events (`L`/`S`/`E` lines the harness writes) carried core, port, offset
and shape, but **no source location**, so no later stage could name a call.

Fixed in `kernel_trace.cpp`:
- The harness's `aie::load_buffer` / `store_buffer` / `release` / `commit` take
  a trailing defaulted `std::source_location`. The body is compiled under
  `#line <bodyLine> "<srcName>"`, so `at.line()` is the user's line. `g_line`
  is set per call and written into every `E` record. `Port::live_line` keeps
  the line of each held block, so `finish()` reports a never-released block at
  the load that **took** it.
- `Event::line` is parsed as the last field of `L`/`S` and the field after `n`
  in `E` (vector reports write `0`).
- `collectHarnessErrors` prints `file:line: port block N on core (..): msg [k
  occurrences] in `<call text>``. `callText()` extracts the call on that line
  for that port (multi-line calls collapse to one line; comments are stripped).
- `declaredShapeHints` now blames per **call line**: a call whose offset and
  shape ranks differ, or whose rank differs from the declaration while other
  calls on the port match it, gets `file:<call line>: port: hint: `<call>` ...`.
  Only when every call agrees, or a block is larger than a declared dim, does it
  blame the declaration (`file:<decl line>: ...`). It then adds one call and
  core as evidence (`e.g. line 20 in `...` on core (0,0) takes 16 in dim 2
  (channels) at offset 0, the declared extent is 4`).

Before:
    a: hint: its load/store passes an offset of 4 and a block shape of 3 coordinates -- the call
       has an extra offset coordinate (the port declares `aie::shape<230,230,4>` (line 3), 3-D)
After:
    aiehlcpm_conv2d.cc:19: a: hint: `aie::load_buffer(a, {oh0 * S, ow0 * S, 0, 0}, {ih_t, iw_t, C})`
       passes an offset of 4 and a block shape of 3 coordinates -- the offset has an extra coordinate
       (the port is 3-D, `aie::shape<230,230,4>` at line 3)

Limit: lines are those of the text the analysis sees. A kernel spliced in by
`inlineSourceIncludes` (opaque, no `#line`) reports lines of the spliced text.
Regression: `never released: the error names the load ...`, plus the
`store on an input port` and `hint:` cases now check `case.cc:<line>:`.

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
