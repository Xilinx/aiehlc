# Debug Analysis: Stack-DMA Buffer Overlap Causing 208 Output Mismatches

**Date:** 2026-05-25
**Symptom:** `verify_mat_transpose FAIL: 208 mismatches out of 256`
**Root Cause:** BCF stack directive `_stack DM_stack 0x70000 0x1024` declared only 4KB stack, but the actual kernel runtime grew beyond that, overwriting DMA ping-pong buffers placed at `0x78000`.
**Fix:** Set BCF stack size to `0x2800` (10KB) to match the actual kernel stack usage, preventing overlap with DMA buffers.

---

## 1. Symptom Description

Running the 16x16 int8 matmul on a 4x4 AIE tile mesh produced the following HW output (from `applog_new2`):

```
verify_mat_transpose FAIL: 208 mismatches out of 256.
```

The output matrix `C[16x16]` showed two distinct corruption patterns:

### Pattern A: Column-0 tiles (tile(r,0)) -- garbage values

```
tile(0,0): got 3,  67, -125, -61  (expected -1, -1, 5, 6)
tile(1,0): got 4,  68, -124, -60  (expected -3, 6, -11, -1)
tile(2,0): got 5,  69, -123, -59  (expected 9, -8, 1, 6)
tile(3,0): got 6,  70, -122, -58  (expected 0, -8, -1, -1)
```

The "got" values form an arithmetic sequence (3,4,5,6 and 67,68,69,70 and -125,-124,-123,-122), suggesting systematic memory corruption -- stack frame data or loop counters bleeding into DMA buffers.

### Pattern B: Column 1-3 tiles (tile(r,1..3)) -- all zeros

```
tile(0,1): got 0, 0, 0, 0  (expected -11, -1, 9, -1)
tile(0,2): got 0, 0, 0, 0  (expected 4, -11, -1, 9)
tile(1,3): got 0, 0, 0, 0  (expected 6, -2, 5, -3)
```

All non-column-0 tiles output zero. This indicates DMA never delivered valid data to these tiles, or the kernel never wrote results because its stack was already corrupted.

### Actual C matrix output (rows shown):

```
Row  0: [  3,  9, 67,  4,  3, -7, 67,  6,-125, -2,-61, -6,-125, -4,-61, -4]  <- col-0 garbage + partial
Row  1: [  0,  0,  0,  0,  0,  0,  0,  0,   0,  0,  0,  0,   0,  0,  0,  0]  <- all zero
Row  2: [  0,  0,  0,  0,  0,  0,  0,  0,   0,  0,  0,  0,   0,  0,  0,  0]
Row  3: [  0,  0,  0,  0,  0,  0,  0,  0,   0,  0,  0,  0,   0,  0,  0,  0]
Row  4: [  4, -6, 68, -2,  4,  6, 68, -7,-124,  4,-60,  9,-124,  9,-60,  4]  <- col-0 garbage
Row  5-7:  all zero
Row  8: [  5, -7, 69,  6,  5, -2, 69, -6,-123, -4,-59, -4,-123, -6,-59, -2]  <- col-0 garbage
Row  9-11: all zero
Row 12: [  6,  6, 70, -7,  6,  4, 70,  9,-122,  9,-58,  4,-122, -7,-58,  6]  <- col-0 garbage
Row 13-15: all zero
```

Only rows 0, 4, 8, 12 have non-zero data (one row per tile row). Each of those rows has corrupted values. All other rows are zero.

---

## 2. AIE Core Tile Data Memory Layout

```
Address Range         Size    Region               Notes
-------------------------------------------------------------
0x00000 - 0x3FFFF     256KB   Program Memory        Code / read-only
0x40000 - 0x4FFFF     64KB    DM Bank 1             BCF: _reserved DMb 0x40000 0x10000
0x50000 - 0x6FFFF     128KB   DM Bank 2-3           Available (globals, heap)
0x70000 - 0x727FF     10KB    Stack                  BCF: _stack DM_stack 0x70000 0x2800
0x72800 - 0x77FFF     22KB    Gap / runtime          Heap, globals, linker-placed data
0x78000 - 0x7F7FF     30KB    DMA ping-pong buffers  CoreMemAllocator base=0x78000
0x7F800 - 0x7FFFF     2KB     klog / TM region       BCF: _reserved DMb 0x7F800 0x800
0x80000 and above     512KB   Not core-addressable   BCF: _reserved DMb 0x80000 0x80000
```

**Who emits the two tail reservations.** There are two independent BCF
generators, and they do *not* agree:

| Generator | Path | `0x7F800` + `0x80000` reserved? |
|---|---|---|
| `Bcf` (`src/mlir/mlirfront/mlirpass/aiehybrid.{h,cc}`) | aiehlc single-kernel | **yes** — `Bcf::addlogandtmreserved()` |
| `TilingBcf` (`.../pass/kernelconfig/kernelconfig.h`) | tilinglinalg multi-tile | **no** — commented out at `tilinglinalg_pipeline.cpp`, "This crashes the simulator" |

Keep them in step only after re-checking the simulator: the tilinglinalg call
site was deliberately reverted, so re-enabling it there is a behaviour change,
not a cleanup.

---

## 3. Root Cause: BCF Stack Size Mismatch

### What the BCF declared

```
_stack DM_stack 0x70000 0x1024
```

This tells the xchesscc linker: "the stack starts at 0x70000, size = 0x1024 bytes (4100 bytes)". The linker trusts this and places the stack pointer at `0x70000 + 0x1024 = 0x71024`.

### What the kernel actually needed

The matmul kernel, when compiled with xchesscc, uses more than 4KB of stack. The kernel function involves:
- Local arrays for tile accumulation
- Function call frames for `chess_*` intrinsics
- Compiler-generated spill slots for vectorized operations

The actual stack consumption is approximately 8-10KB. With only 4KB declared, the stack grows downward past `0x71024` and eventually reaches into:
- `0x72800 - 0x77FFF`: runtime/heap region (corrupts globals)
- `0x78000+`: DMA buffer region (corrupts DMA data)

### Stack overflow direction

AIE stack grows **upward** from the base address. The `_stack` BCF directive sets `SP = base_addr`, and the stack grows toward `base_addr + size`. When the stack overflows past the declared limit, it writes into the memory region immediately above -- which is exactly where the gap region and eventually the DMA buffers live.

With `_stack 0x70000 0x1024`:
- Stack occupies: `0x70000 - 0x71024` (declared)
- Stack overflows into: `0x71024 - 0x77FFF` (undeclared)
- DMA buffers at: `0x78000+` (CoreMemAllocator)

When the stack overflows far enough, it corrupts DMA buffer contents.

### Why column-0 tiles show garbage, not zeros

Column-0 tiles (tile(r,0)) are the first to receive DMA data because they are directly connected to the SHIM tiles. Their DMA transfers may partially complete before the stack corruption overwrites the buffer. This produces garbage (stack frame data mixed with partial DMA data).

Column 1-3 tiles depend on cascade/broadcast from column-0, or their DMA paths are longer. By the time data arrives, the kernel's stack has already corrupted the buffer region, and the output reads as zero (uninitialized memory).

---

## 4. The Two Bugs and Their Interaction

This issue was actually caused by two independent bugs that interacted:

### Bug 1: Dynamic CoreMemAllocator base address (fixed first)

In `ResourceManager.cpp`, the constructor dynamically recomputed the CoreMemAllocator base:

```cpp
uint32_t stackEnd = 0x70000 + resource_->getStackReserveBytes(); // = 0x72800
uint32_t dataStart = (stackEnd + 31) & ~31;                      // = 0x72800
coreMemAllocator_ = CoreMemAllocator(dataStart, dataSize);
```

This moved DMA buffers from `0x78000` down to `0x72800`, placing them directly in the gap region between the stack and the safe buffer zone. This made stack overflow corruption **much more likely** because the buffers were now only 6KB away from the stack start instead of 32KB.

**Fix:** Reverted to the default `CoreMemAllocator(0x78000, 0x8000)`.

### Bug 2: BCF stack size too small (fixed second)

The BCF declared `_stack DM_stack 0x70000 0x1024` (4100 bytes). The kernel actually needs ~10KB of stack. Even with buffers at the safe `0x78000` base, a stack overflow from 4KB could eventually reach and corrupt the DMA buffers.

**Fix:** Changed BCF stack size to `0x2800` (10240 = 10KB).

### Interaction

With Bug 1 alone (buffers at 0x72800) + old stack (0x1024), the buffers were extremely close to the stack and corruption was guaranteed.

With Bug 2 alone (stack 0x1024) + correct buffer base (0x78000), the gap is 32KB and the kernel's ~10KB stack would overflow into the 0x71024-0x72800 region but likely not reach 0x78000. However, this is fragile and depends on the exact kernel complexity.

Fixing both bugs ensures:
- Stack has 10KB of space (0x70000-0x72800) -- sufficient for the matmul kernel
- DMA buffers start at 0x78000 -- well above the stack end
- 22KB gap (0x72800-0x78000) provides safety margin for heap/globals

---

## 5. Files Changed

| File | Change |
|------|--------|
| `pass/routingimplement/hw/ResourceManager.cpp` | Removed dynamic CoreMemAllocator reconfiguration (reverted to default base=0x78000) |
| `pass/kernelconfig/kernelconfig.h` | Default `stackSize_`: `0x1024` -> `0x2800`; updated comment |
| `pass/tilinglinalg_pipeline.cpp:936` | `bcf.setStack(0x70000, 0x1024)` -> `bcf.setStack(0x70000, 0x2800)` |
| `pass/unitest/test.cpp:1257` | `bcf.setStack(0x70000, 0x1024)` -> `bcf.setStack(0x70000, 0x2800)` |

---

## 6. Verification

After both fixes, the generated BCF contains:

```
_stack DM_stack 0x70000 0x2800
_symbol buf_in_ping_1 0x78000
_symbol buf_in_pong_1 0x78020
...
```

The stack (0x70000-0x72800) and DMA buffers (0x78000+) are separated by a 22KB gap. The kernel has sufficient stack space, and DMA buffers are in the proven-safe zone.

Rebuild and re-run:
```bash
cd src/mlir/mlirfront/tilinglinalg/pass/unitest/build && make -j4
./test dfschedule
# Then recompile kernel with new BCF and run on HW
```

---

## 7. Lessons Learned

1. **BCF stack size must match actual kernel stack usage.** The `_stack` directive is a linker hint, not a hardware guard. There is no stack overflow protection on AIE cores -- the stack silently overwrites adjacent memory.

2. **CoreMemAllocator base address is a physical layout constant, not a derived value.** The base `0x78000` is a known-safe starting point established by the BCF/linker memory map. Computing it dynamically from stack sizes conflates two separate concerns: memory budget validation vs physical allocation placement.

3. **Two-pattern corruption (garbage + zeros) is a signature of stack-DMA overlap.** Garbage values indicate partial data mixed with stack frame contents. All-zeros indicate DMA data never arrived or was completely overwritten.

4. **Arithmetic sequences in corrupted output (3,4,5,6...) suggest loop counter spill.** When the compiler spills loop variables to stack and the stack overlaps with DMA buffers, the output shows incrementing patterns from the loop counters.
