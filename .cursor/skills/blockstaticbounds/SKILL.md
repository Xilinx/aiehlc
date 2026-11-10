---
name: blockstaticbounds
description: A block-load kernel fails `[KernelAnalysis] ERROR ... data used without being loaded (index past the block shape)` and you need WHICH constant causes it (e.g. `oc_t = 15` while the compute loops walk 16 channels). Reading / extending the static bounds analysis (`src/llvm/convautotile/kernel_static.{h,cpp}`) -- `static analysis:` lines, `out-of-block access (static analysis)`, `cause (static analysis)`, `possible out-of-block access` notes, `N block access(es) checked, M not evaluable`.
---

# Static block bounds: the constant behind "data used without being loaded"

## Symptom

```
[KernelAnalysis] ERROR: conv2d_spatial: the kernel accesses memory outside the block
  load_buffer/store_buffer returned at simpleconv2dkernel_port.cc:29 on core (0,0):
  data used without being loaded (index past the block shape)
```

That comes from the x86 trace (ASan). It gives the **line** and stops at the first
fault. `staticBlockBounds()` (kernel_static.cpp) runs right after it inside
`traceKernel` / `traceLogical` and adds the **why**:

```
    static analysis: `w[(nb * 8 + n) * TAPS + kh * KW * C + byte]` reads element 3135 of block `w`
      (2940 elements, shape (15,7,7,4) = aie::load_buffer(win_b, ..., {oc_t, KH, KW, C}) [WEIGHTS], line 18)
      at kh=6, kb=3, nb=1, k=7, n=7: element 3135 is (15,6,6,3), dim 0 (output channels) index 15 is past
      its extent 15 (196 elements past the end); it is driven by `nb < 2` (line 23), `n < 8` (line 25);
      neither the index nor these loops depend on oc_t: they walk 16 entries of dim 0 whatever oc_t is
ERROR: out-of-block access (static analysis) at ...:74: `aie::store_v(o + 96, ...)` writes element 2947 ...
ERROR: cause (static analysis): `oc_t = 15` (...:8) sizes the block of the access at line 29, whose index
       never reads oc_t -- with oc_t = 16 every analysed block access fits, including ... lines 29, 74
```

The fix in `simpleconv2dkernel_port.cc` is `oc_t = 16`. Its compute hardcodes 16
output channels (`nb < 2` x `n < 8`, 128-byte `store_v` groups). To support other
widths, bound those loops by `oc_t` instead.

## How it decides (do not loosen these)

- **Exact enumeration, not intervals.** Each needed loop variable (and `program_id`)
  is enumerated. Intervals would make `owB = pass*16 + (pass ? 4 : 8)` span [4,24]
  and raise a false overflow on the correct kernel.
- **A cause must be decoupled.** A literal constant counts as the cause only if it
  is read by the block **shape** but not by the overflowing **index** or its loops,
  *and* changing it alone makes every analysed access fit. `ow_t` (block width and
  row pitch both) would "fit" at 29 or 47 just by adding slack. That once
  mis-blamed the `load too small` / `off = kb * 8` regression cases, so such
  constants are excluded and no cause line is printed.
- **Unknown means skipped, never guessed, and the reason is printed.** Data-dependent
  indices, unknown calls, reassigned names (`x = aie::load_buffer(...)` again, a
  running `kk++` counter), multi-dim local arrays and pointees of unknown size are
  skipped, each with a `not evaluable at <file>:<line>` note.
- **Bytes.** Offsets are summed in bytes (each pointer's pointee size), so
  `(const int8_t *)a` over an int16 block is exact.
- **Errors vs notes.** Static findings are errors only when the trace also faulted.
  Otherwise they are `possible ...` notes, because an unmodelled guard may stop the
  access from running.

## Reading the other messages

| Message | Meaning |
|---|---|
| `note: static bounds analysis: not evaluable at k.cc:69 \`w[oc * TAPS + kk]\`: \`kk\` is reassigned at line 67` | skipped access, with the reason; nothing was guessed |
| `` `w` is used in a form the analysis does not model: `use(w, s);` `` | a block pointer escapes (helper, `&w`, `*(w + e)`); reads through it are invisible |
| `static analysis: cannot explain line L: ...` (appended to an ASan error) | the trace faulted on an access the analysis skipped |
| `static analysis: no block access recognized at line L` | the fault is through something not modelled as an access |
| `static analysis DISAGREES: it evaluated ... and found it inside its block` | **the static model is wrong for this kernel** (alias, cast, escape). Trust ASan; fix the model; add a regression case |
| `reads byte 64 of block \`a\` (64 bytes, 2 B per element, accessed through \`int8_t\`)` | byte units: the pointer type differs from the port's element type |

## Verify after a change

```bash
src/llvm/convautotile/unitest/run.sh                    # testStaticBounds (C++, fast)
python3 src/llvm/convautotile/unitest/kernel_trace_cases.py build/aiehlc-convtile
./build/aiehlc-convtile trace example/tileprogram/ccode/simpleconv2dkernel_port.cc
```

The baseline (`oc_t = 16`) must print `11 block access(es) checked` and no
`possible` note: any note there is a false positive.

## Sandbox note

The harness and `aiehlc.sh` write temp files under `$TMPDIR` (default `/tmp`). A
read-only `/tmp` used to fail with `kernel analysis: cannot create a temp
directory` / `mktemp: failed to create file via template '/tmp/aiehlc_err...'`.
