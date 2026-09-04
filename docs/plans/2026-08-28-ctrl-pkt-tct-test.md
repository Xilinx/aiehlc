# Control-Packet TCT Test API Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a single AIE-runtime API `__Runtime_ctrl_pkt_test` that takes a transaction buffer (`XAie_TxnCmd` stream), converts it to control packets, self-routes to a destination tile's CTRL port, arms an S2MM return-path TCT into DDR, sends, waits on the TCT, and reports wall-time/throughput.

**Architecture:** One public orchestrator function plus `static` helpers in `src/mlir/runtime/aie_runtime.c`, each under the project's 200-line limit. Pure-math helpers (txn parse, packetize reuse) are unit-tested host-side with a standalone g++ harness; XAie-dependent helpers (routing, S2MM arm, poll) are compile-checked via the existing host build (`script/hostcompile.sh`) and exercised on HW via `apppaltest.py`.

**Tech Stack:** C (C++17-compiled), AMD XAie driver (`xaiengine`), existing runtime primitives `__Runtime_ctrl_pktize` / `__Runtime_ctrl_push` / `__Runtime_alloc_buffer`.

**Design doc:** `docs/plans/2026-08-28-ctrl-pkt-tct-test-design.md`

**Project rules to honor:**
- "never do: api function > 200 lines" — keep every function short.
- "Process transparent rule": after each task list files changed.
- "Document rule": update `CLAUDE.md`/design doc after changes.

**Build/verify notes:**
- Shell profile prints env noise; redirect build output to `/tmp/claude/*.log` and Read it.
- `git`/build may need `dangerouslyDisableSandbox: true` (bwrap mount error in sandbox).
- Runtime is compiled by `script/hostcompile.sh` (aarch64 g++, XAie includes). The MLIR `pass/unitest` does NOT link the runtime — do not add runtime-C tests there.

---

## Task 1: Result struct + public prototype in the header

**Files:**
- Modify: `src/mlir/runtime/aie_runtime.h` (append after the control-packet block, ~line 701, before `__Runtime_sync_for_dev`)

**Step 1: Add the struct + prototype + doc comment**

Insert after the `__Runtime_ctrl_push` declaration (aie_runtime.h:701):

```c
// ---------------------------------------------------------------------------
// Control-packet TCT (task-completion-token) test API.
// Takes a transaction buffer (XAie_TxnCmd stream), converts its register-write
// ops into control packets, self-programs the SHIM->dest CTRL route plus the
// dest->SHIM S2MM return route, pushes the packet, polls a DDR token for the
// TCT, and reports host wall-time + throughput. Single self-contained API.
// ---------------------------------------------------------------------------
typedef struct {
    AieRC    rc;              // XAIE_OK on success
    uint32_t pkt_words;       // control-packet words actually sent
    uint32_t payload_bytes;   // register-write payload bytes (excl. headers)
    uint64_t elapsed_us;      // host wall time: push-start -> TCT-done
    double   throughput_mbps; // payload_bytes / elapsed_us (0 if elapsed_us==0)
    uint32_t tct_value;       // token word observed in DDR
} __Runtime_CtrlPktTestResult;

// Run one control-packet TCT test. `txn`/`txn_count` is a raw XAie_TxnCmd
// stream; WRITE/BLOCKWRITE ops are packetized to `dest_(col,row)` CTRL via
// `shim_col` MM2S ch `mm2s_ch` with stream id `ctrl_stream_id`; the TCT returns
// on `shim_col` S2MM ch `s2mm_ch`. dev must be partitioned first.
__Runtime_CtrlPktTestResult
__Runtime_ctrl_pkt_test(XAie_DevInst *dev,
                        const struct XAie_TxnCmd *txn, uint32_t txn_count,
                        uint8_t shim_col, uint8_t dest_col, uint8_t dest_row,
                        uint8_t ctrl_stream_id, uint8_t mm2s_ch, uint8_t s2mm_ch);
```

**Step 2: Verify header self-consistency**

Confirm `xaie_txn.h` (providing `struct XAie_TxnCmd`) is reachable via the existing `#include <xaiengine.h>` in `aie_runtime.h`. Run:
`grep -n "xaiengine" src/mlir/runtime/aie_runtime.h`
Expected: an include of the XAie umbrella header. If `struct XAie_TxnCmd` is not transitively included, add `#include <xaiengine/xaie_txn.h>` near the other XAie includes.

**Step 3: Commit**

```bash
git add src/mlir/runtime/aie_runtime.h
git commit -m "feat(runtime): declare __Runtime_ctrl_pkt_test API + result struct"
```

---

## Task 2: Pure txn-parse helper + standalone unit test (TDD)

The txn parser is pure math (no XAie device), so test it first with a standalone host g++ harness.

**Files:**
- Create: `src/mlir/runtime/tests/test_ctrl_txn.c`
- Create: `src/mlir/runtime/tests/ctrl_txn_parse.h` (extracted pure helper, included by both the test and aie_runtime.c)

**Step 1: Write the failing test**

Create `src/mlir/runtime/tests/test_ctrl_txn.c`:

```c
#include "ctrl_txn_parse.h"
#include <assert.h>
#include <stdio.h>
#include <string.h>

/* Minimal standalone copies of the XAie txn types the parser needs, so this
 * test builds with plain g++/gcc (no XAie/Vitis). Field layout mirrors
 * xaie_txn.h. */
typedef enum { XAIE_IO_WRITE = 0, XAIE_IO_BLOCKWRITE = 1,
               XAIE_IO_CUSTOM_OP_TCT = 128 } TestOpcode;
struct TestTxnCmd { TestOpcode Opcode; unsigned Mask; unsigned long long RegOff;
                    unsigned Value; unsigned long long DataPtr; unsigned Size; };

int main(void) {
    struct TestTxnCmd cmds[3] = {
        { XAIE_IO_WRITE, 0, 0x1000, 0xAABBCCDD, 0, 0 },
        { XAIE_IO_WRITE, 0, 0x1004, 0x11223344, 0, 0 },
        { XAIE_IO_CUSTOM_OP_TCT, 0, 0, 0, 0, 0 },
    };
    unsigned data[8]; unsigned tile_addr = 0xFFFF; int has_tct = -1;
    unsigned n = ctrl_txn_collect(cmds, 3, sizeof(struct TestTxnCmd),
                                  &tile_addr, data, 8, &has_tct);
    assert(n == 2);
    assert(tile_addr == 0x1000);
    assert(data[0] == 0xAABBCCDD);
    assert(data[1] == 0x11223344);
    assert(has_tct == 1);
    printf("test_ctrl_txn: PASS (%u words, addr=0x%x, tct=%d)\n", n, tile_addr, has_tct);
    return 0;
}
```

**Step 2: Run it to verify it fails (no header/function yet)**

```bash
mkdir -p /tmp/claude
g++ -std=c++17 -I src/mlir/runtime/tests \
    src/mlir/runtime/tests/test_ctrl_txn.c -o /tmp/claude/test_ctrl_txn \
    > /tmp/claude/t2.log 2>&1; echo EXIT=$?
```
Expected: FAIL — `ctrl_txn_parse.h: No such file` or `ctrl_txn_collect` undefined.

**Step 3: Write the minimal pure helper**

Create `src/mlir/runtime/tests/ctrl_txn_parse.h`:

```c
#ifndef CTRL_TXN_PARSE_H
#define CTRL_TXN_PARSE_H
#include <stddef.h>
#include <stdint.h>

/* Opcode values must match xaie_txn.h XAie_TxnOpcode. */
#ifndef CTRL_TXN_OP_WRITE
#define CTRL_TXN_OP_WRITE       0u
#define CTRL_TXN_OP_BLOCKWRITE  1u
#define CTRL_TXN_OP_TCT         128u
#endif

/*
 * Collect register-write words from a raw txn stream into `data`.
 * `cmds` points to `count` records each `stride` bytes wide, laid out as
 * { uint32 Opcode; uint32 Mask; uint64 RegOff; uint32 Value; uint64 DataPtr;
 *   uint32 Size; } (matches struct XAie_TxnCmd on LP64). `stride` lets callers
 * pass the real sizeof(struct XAie_TxnCmd) so padding differences don't matter.
 *
 * Returns number of words written to `data` (capped at `cap`); sets
 * `*tile_addr` to the first WRITE RegOff (low 20 bits) and `*has_tct` to 1 if a
 * TCT op was seen, else 0.
 */
static inline uint32_t ctrl_txn_collect(const void *cmds, uint32_t count,
                                        size_t stride, uint32_t *tile_addr,
                                        uint32_t *data, uint32_t cap,
                                        int *has_tct) {
    uint32_t n = 0; int first = 1; if (has_tct) *has_tct = 0;
    const unsigned char *base = (const unsigned char *)cmds;
    for (uint32_t i = 0; i < count; i++) {
        const unsigned char *rec = base + (size_t)i * stride;
        uint32_t opcode; uint64_t regoff; uint32_t value;
        /* Field offsets per the documented layout above. */
        memcpy(&opcode, rec + 0, sizeof(opcode));
        memcpy(&regoff, rec + 8, sizeof(regoff));
        memcpy(&value,  rec + 16, sizeof(value));
        if (opcode == CTRL_TXN_OP_WRITE || opcode == CTRL_TXN_OP_BLOCKWRITE) {
            if (first) { if (tile_addr) *tile_addr = (uint32_t)(regoff & 0xFFFFFu); first = 0; }
            if (n < cap) data[n] = value;
            n++;
        } else if (opcode == CTRL_TXN_OP_TCT) {
            if (has_tct) *has_tct = 1;
        }
    }
    if (n > cap) n = cap;
    return n;
}
#endif /* CTRL_TXN_PARSE_H */
```

**Step 4: Run the test to verify it passes**

```bash
g++ -std=c++17 -I src/mlir/runtime/tests \
    src/mlir/runtime/tests/test_ctrl_txn.c -o /tmp/claude/test_ctrl_txn \
    > /tmp/claude/t2.log 2>&1 && /tmp/claude/test_ctrl_txn >> /tmp/claude/t2.log 2>&1; echo EXIT=$?
cat /tmp/claude/t2.log
```
Expected: `test_ctrl_txn: PASS (2 words, addr=0x1000, tct=1)` and `EXIT=0`.

> NOTE on field offsets: the memcpy offsets (0/8/16) assume LP64 layout of
> `struct XAie_TxnCmd` { XAie_TxnOpcode(enum→4B) Opcode; u32 Mask; u64 RegOff;
> u32 Value; u64 DataPtr; u32 Size; } with natural 8-byte alignment before RegOff
> and DataPtr. When wiring into aie_runtime.c (Task 4) do NOT use raw offsets —
> access the real struct fields directly. This standalone helper's offset-based
> reader exists ONLY so the pure test builds without XAie headers. Keep the two
> in sync via the shared logic, not shared binary layout.

**Step 5: Commit**

```bash
git add src/mlir/runtime/tests/ctrl_txn_parse.h src/mlir/runtime/tests/test_ctrl_txn.c
git commit -m "test(runtime): pure txn->reg-word parser with standalone unit test"
```

---

## Task 3: Packetize integration check (reuse __Runtime_ctrl_pktize)

Confirm the parsed words + `__Runtime_ctrl_pktize` produce the expected control-packet word count so Task 4 wiring is correct. This is a reasoning/verification task against existing code (aie_runtime.c:3016).

**Files:**
- Reference only: `src/mlir/runtime/aie_runtime.c:3016-3042`

**Step 1: Compute expected packet size**

For `nwords` reg words, `__Runtime_ctrl_pktize` emits, per <=4-word chunk,
`2 headers + chunk_size` words. For 2 words: one chunk of 2 → `2 + 2 = 4` words.
Record this expected value (4) for the Task 6 wiring assertion / HW log check.

**Step 2: (No code) — document the contract in the design doc**

Append a one-line note to `docs/plans/2026-08-28-ctrl-pkt-tct-test-design.md`
under "Data Flow": `pkt_words = sum_over_chunks(2 + chunk_size)`.

**Step 3: Commit**

```bash
git add docs/plans/2026-08-28-ctrl-pkt-tct-test-design.md
git commit -m "docs: record ctrl-packet word-count contract"
```

---

## Task 4: Implement static helpers in aie_runtime.c

**Files:**
- Modify: `src/mlir/runtime/aie_runtime.c` (add after `__Runtime_ctrl_push`, ~line 3096)

**Step 1: Add txn-collect + route + S2MM-arm + poll helpers**

Insert the following `static` helpers. Each stays well under 200 lines. Use the
real `struct XAie_TxnCmd` fields (NOT raw offsets) — the logic mirrors
`ctrl_txn_collect` in ctrl_txn_parse.h.

```c
/* Collect WRITE/BLOCKWRITE data words + base tile addr from a txn stream.
 * Mirrors tests/ctrl_txn_parse.h ctrl_txn_collect but uses the real struct. */
static uint32_t rt_ctrl_txn_collect(const struct XAie_TxnCmd *txn, uint32_t count,
                                    uint32_t *tile_addr, uint32_t *data,
                                    uint32_t cap, int *has_tct) {
    uint32_t n = 0; int first = 1; if (has_tct) *has_tct = 0;
    for (uint32_t i = 0; i < count; i++) {
        XAie_TxnOpcode op = txn[i].Opcode;
        if (op == XAIE_IO_WRITE || op == XAIE_IO_BLOCKWRITE) {
            if (first) { if (tile_addr) *tile_addr = (uint32_t)(txn[i].RegOff & 0xFFFFFu); first = 0; }
            if (n < cap) data[n] = txn[i].Value;
            n++;
        } else if (op == XAIE_IO_CUSTOM_OP_TCT) {
            if (has_tct) *has_tct = 1;
        }
    }
    if (n > cap) n = cap;
    return n;
}

/* Program forward CTRL route SHIM(shim_col,0) MM2S -> dest CTRL port, and the
 * dest -> SHIM S2MM return route for the TCT token. Returns first non-OK rc.
 * NOTE: exact hop sequence depends on the physical dest location; this uses a
 * straight vertical NORTH climb in the dest column then into the CTRL port,
 * mirroring the aie_control.cpp CTRL-sink pattern (DONOT_DROP_HEADER). */
static AieRC rt_ctrl_route_setup(XAie_DevInst *dev, uint8_t shim_col,
                                 uint8_t dest_col, uint8_t dest_row,
                                 uint8_t stream_id, uint8_t s2mm_ch) {
    AieRC rc = XAIE_OK;
    XAie_LocType shim = XAie_TileLoc(shim_col, 0);
    XAie_LocType dst  = XAie_TileLoc(dest_col, dest_row);
    /* Forward: SHIM south-in from DMA (MM2S) up to dest, terminate at CTRL. */
    rc = XAie_StrmConnCctEnable(dev, shim, DMA, 0, NORTH, 0);
    /* Intermediate vertical hops dest_col rows 1..dest_row-1 */
    for (uint8_t r = 1; rc == XAIE_OK && r < dest_row; r++)
        rc = XAie_StrmConnCctEnable(dev, XAie_TileLoc(dest_col, r), SOUTH, 0, NORTH, 0);
    if (rc == XAIE_OK)
        rc = XAie_StrmPktSwSlaveSlotEnable(dev, dst, SOUTH, 0, 0,
                 (XAie_Packet){.PktId = stream_id, .PktType = 0}, 0x1F, 0, 0);
    if (rc == XAIE_OK)
        rc = XAie_StrmPktSwMstrPortEnable(dev, dst, CTRL, 0,
                 XAIE_SS_PKT_DONOT_DROP_HEADER, 0, 0x1);
    /* Return: dest -> SHIM S2MM for the TCT token. */
    for (int r = (int)dest_row; rc == XAIE_OK && r > 0; r--)
        rc = XAie_StrmConnCctEnable(dev, XAie_TileLoc(dest_col, (uint8_t)r), NORTH, 1, SOUTH, 1);
    if (rc == XAIE_OK)
        rc = XAie_StrmConnCctEnable(dev, shim, SOUTH, 1, DMA, s2mm_ch);
    if (rc != XAIE_OK)
        printf("[aie_runtime] ctrl_route_setup ERROR rc=%d shim=%u dest=(%u,%u)\n",
               (int)rc, (unsigned)shim_col, (unsigned)dest_col, (unsigned)dest_row);
    return rc;
}

/* Arm a SHIM S2MM BD writing the TCT token into `token_buf` (DMA buffer). */
static AieRC rt_tct_s2mm_arm(XAie_DevInst *dev, uint8_t shim_col, uint8_t s2mm_ch,
                             uint32_t *token_buf, int32_t bd_id) {
    XAie_LocType loc = XAie_TileLoc(shim_col, 0);
    uint64_t offset = 0;
    XAie_MemInst *mem = __vaddr_to_mem_offset(token_buf, &offset);
    if (!mem) { printf("[aie_runtime] tct_s2mm_arm ERROR: token_buf not a DMA buffer\n"); return XAIE_ERR; }
    uint64_t dev_addr = XAie_MemGetDevAddr(mem) + offset;
    XAie_DmaDesc desc; AieRC rc = XAie_DmaDescInit(dev, &desc, loc);
    if (rc == XAIE_OK) rc = XAie_DmaSetAddrLen(&desc, dev_addr, (uint32_t)sizeof(uint32_t));
    if (rc == XAIE_OK) rc = XAie_DmaEnableBd(&desc);
    if (rc == XAIE_OK) rc = XAie_DmaSetAxi(&desc, 0, 16, 0, 0, 0);
    if (rc == XAIE_OK) rc = XAie_DmaWriteBd(dev, &desc, loc, (uint8_t)bd_id);
    if (rc == XAIE_OK) rc = XAie_DmaChannelPushBdToQueue(dev, loc, s2mm_ch, DMA_S2MM, (uint8_t)bd_id);
    if (rc == XAIE_OK) rc = XAie_DmaChannelEnable(dev, loc, s2mm_ch, DMA_S2MM);
    if (rc != XAIE_OK) printf("[aie_runtime] tct_s2mm_arm ERROR rc=%d\n", (int)rc);
    return rc;
}

/* Poll the DDR token buffer (with S2MM pending-BD backstop) for the TCT. */
static uint32_t rt_tct_poll(XAie_DevInst *dev, uint32_t *token_buf,
                            uint8_t shim_col, uint8_t s2mm_ch) {
    XAie_LocType loc = XAie_TileLoc(shim_col, 0);
    const uint32_t max_iters = 50000000u;
    for (uint32_t iter = 0; iter < max_iters; iter++) {
        u8 pending = 1;
        (void)XAie_DmaGetPendingBdCount(dev, loc, s2mm_ch, DMA_S2MM, &pending);
        if (pending == 0) break;
    }
    __Runtime_sync_for_cpu(dev, token_buf, sizeof(uint32_t));
    return token_buf[0];
}
```

**Step 2: Compile-check via host build**

```bash
source script/aiehlc.sh --aie-version 5 --runtime-source-file ./example/tileprogram/ccode/simplematmul2.cc > /tmp/claude/t4gen.log 2>&1; echo GEN_EXIT=$?
```
Then confirm `aie_runtime.c` compiled during host codegen:
```bash
grep -iE "Compiling aie_runtime.c|error: |aie_runtime" /tmp/claude/t4gen.log | tail -20
```
Expected: `Compiling aie_runtime.c...` present, no `error:` lines from aie_runtime.c.

> If `XAie_Packet` / `CTRL` / `XAIE_SS_PKT_DONOT_DROP_HEADER` names differ, grep
> the generated `aie_control.cpp` (`src/mlir/mlirfront/tilinglinalg/pass/routingimplement/codegenexample/aie_control.cpp`)
> for the exact spellings and match them.

**Step 3: Commit**

```bash
git add src/mlir/runtime/aie_runtime.c
git commit -m "feat(runtime): ctrl-pkt txn-collect, route-setup, S2MM-arm, TCT-poll helpers"
```

---

## Task 5: Implement the public orchestrator __Runtime_ctrl_pkt_test

**Files:**
- Modify: `src/mlir/runtime/aie_runtime.c` (add after the Task-4 helpers)

**Step 1: Add the public function (thin orchestrator, < 80 lines)**

```c
__Runtime_CtrlPktTestResult
__Runtime_ctrl_pkt_test(XAie_DevInst *dev, const struct XAie_TxnCmd *txn, uint32_t txn_count,
                        uint8_t shim_col, uint8_t dest_col, uint8_t dest_row,
                        uint8_t ctrl_stream_id, uint8_t mm2s_ch, uint8_t s2mm_ch) {
    __Runtime_CtrlPktTestResult res; memset(&res, 0, sizeof(res)); res.rc = XAIE_OK;

    /* 1. parse txn -> reg words */
    uint32_t words[256]; uint32_t tile_addr = 0; int has_tct = 0;
    uint32_t nwords = rt_ctrl_txn_collect(txn, txn_count, &tile_addr, words, 256, &has_tct);
    res.payload_bytes = nwords * (uint32_t)sizeof(uint32_t);

    /* 2. packetize into a DMA buffer */
    uint32_t cap = nwords * 2u + 8u;
    uint32_t *pkt = (uint32_t *)__Runtime_alloc_buffer(dev, cap * sizeof(uint32_t));
    uint32_t *token = (uint32_t *)__Runtime_alloc_buffer(dev, sizeof(uint32_t));
    if (!pkt || !token) { res.rc = XAIE_ERR; goto done; }
    token[0] = 0u;
    res.pkt_words = __Runtime_ctrl_pktize(pkt, cap, ctrl_stream_id, tile_addr, words, nwords);
    if (res.pkt_words == 0u) { res.rc = XAIE_ERR; goto done; }

    /* 3. routing (forward + return) */
    res.rc = rt_ctrl_route_setup(dev, shim_col, dest_col, dest_row, ctrl_stream_id, s2mm_ch);
    if (res.rc != XAIE_OK) goto done;

    /* 4. arm return-path S2MM for TCT */
    res.rc = rt_tct_s2mm_arm(dev, shim_col, s2mm_ch, token, /*bd_id=*/1);
    if (res.rc != XAIE_OK) goto done;

    /* 5. timed push + TCT poll */
#ifndef __AIESIM__
    uint64_t t0 = __Runtime_host_now_counts();
#endif
    res.rc = __Runtime_ctrl_push(dev, shim_col, pkt, res.pkt_words, /*bd_id=*/0, mm2s_ch);
    if (res.rc != XAIE_OK) goto done;
    res.tct_value = rt_tct_poll(dev, token, shim_col, s2mm_ch);
#ifndef __AIESIM__
    uint64_t t1 = __Runtime_host_now_counts();
    uint64_t cps = __Runtime_host_counts_per_sec();
    if (cps) res.elapsed_us = ((t1 - t0) * 1000000ull) / cps;
    if (res.elapsed_us) res.throughput_mbps = (double)res.payload_bytes / (double)res.elapsed_us;
#endif

    printf("[aie_runtime] ctrl_pkt_test: pkt_words=%u payload=%uB elapsed=%lluus tput=%.3fMB/s tct=0x%x\n",
           res.pkt_words, res.payload_bytes, (unsigned long long)res.elapsed_us,
           res.throughput_mbps, res.tct_value);
done:
    if (pkt) __Runtime_free_buffer(dev, pkt);
    if (token) __Runtime_free_buffer(dev, token);
    return res;
}
```

**Step 2: Provide the host-clock helpers if missing**

Check whether host-clock wrappers exist:
```bash
grep -nE "__Runtime_host_now_counts|__Runtime_host_counts_per_sec|XTime_GetTime|COUNTS_PER_SECOND" src/mlir/runtime/aie_runtime.c | head
```
- If present under other names, use those and adjust Step 1.
- If absent, add two tiny static-free wrappers near the top of aie_runtime.c:
```c
uint64_t __Runtime_host_now_counts(void) {
#if defined(XTime_GetTime)
    XTime t; XTime_GetTime(&t); return (uint64_t)t;
#else
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
#endif
}
uint64_t __Runtime_host_counts_per_sec(void) {
#if defined(COUNTS_PER_SECOND)
    return (uint64_t)COUNTS_PER_SECOND;
#else
    return 1000000000ull; /* clock_gettime is ns */
#endif
}
```
And declare both in aie_runtime.h. (Match whatever the existing trace-sync code uses for the host clock — grep first, reuse if found.)

**Step 3: Compile-check via host build**

```bash
source script/aiehlc.sh --aie-version 5 --runtime-source-file ./example/tileprogram/ccode/simplematmul2.cc > /tmp/claude/t5gen.log 2>&1; echo GEN_EXIT=$?
grep -iE "error: |Compiling aie_runtime.c" /tmp/claude/t5gen.log | tail -20
```
Expected: `Compiling aie_runtime.c...`, no `error:` from aie_runtime.c.

**Step 4: Commit**

```bash
git add src/mlir/runtime/aie_runtime.c src/mlir/runtime/aie_runtime.h
git commit -m "feat(runtime): implement __Runtime_ctrl_pkt_test orchestrator"
```

---

## Task 6: Re-run pure unit test + regression + docs

**Files:**
- Modify: `CLAUDE.md` (runtime API note), `docs/plans/2026-08-28-ctrl-pkt-tct-test-design.md`

**Step 1: Re-run the standalone txn parser test**

```bash
g++ -std=c++17 -I src/mlir/runtime/tests src/mlir/runtime/tests/test_ctrl_txn.c \
    -o /tmp/claude/test_ctrl_txn && /tmp/claude/test_ctrl_txn; echo EXIT=$?
```
Expected: `test_ctrl_txn: PASS ...` `EXIT=0`.

**Step 2: Full matmul2 regression gen (ensure nothing else broke)**

```bash
source script/aiehlc.sh --aie-version 5 --runtime-source-file ./example/tileprogram/ccode/simplematmul2.cc > /tmp/claude/t6gen.log 2>&1; echo GEN_EXIT=$?
grep -iE "error: |fail" /tmp/claude/t6gen.log | grep -v "0 error" | tail -20
```
Expected: `GEN_EXIT=0`, no compile errors.

**Step 3: Document the new API**

- Add `__Runtime_ctrl_pkt_test` to any runtime API list in `CLAUDE.md` (Part 1 runtime API bullet) and confirm the design doc's "Files to Change" matches reality.

**Step 4: Commit**

```bash
git add CLAUDE.md docs/plans/2026-08-28-ctrl-pkt-tct-test-design.md
git commit -m "docs: document __Runtime_ctrl_pkt_test runtime API"
```

---

## Task 7 (optional, HW): live device run

**Only if a board session is available.** Add a tiny caller in a scratch host
`.cc` (or the debug harness) that builds a 2-op `XAie_TxnCmd[]` and calls
`__Runtime_ctrl_pkt_test`, then run:

```bash
python3 script/test/apppaltest.py -y -nonreboot > /tmp/claude/hwrun.log 2>&1; echo EXIT=$?
grep -iE "ctrl_pkt_test|device_teardown done|AIE ERROR" /tmp/claude/hwrun.log
```
Expected: a `ctrl_pkt_test: ... tct=0x...` line with non-zero `tct` and
`device_teardown done`; no `AIE ERROR`.

---

## Verification Summary

| Task | Verification |
|------|--------------|
| 1 | header grep / self-consistency |
| 2 | standalone g++ unit test PASS |
| 3 | word-count contract documented |
| 4 | host build compiles aie_runtime.c, no errors |
| 5 | host build compiles orchestrator, no errors |
| 6 | unit test re-PASS + matmul2 regression GEN_EXIT=0 |
| 7 | (optional) HW run: tct!=0, device_teardown done |

## Open Risks / Adjustments During Execution

- **Exact XAie route API names** (`XAie_Packet`, `CTRL`, `XAIE_SS_PKT_DONOT_DROP_HEADER`, port enums) must be cross-checked against the generated `aie_control.cpp` — adjust spellings in Task 4 if the build complains.
- **Hop topology** in `rt_ctrl_route_setup` is a straight-column climb; real dest tiles off the shim column need an added horizontal (EAST/WEST) leg. If `dest_col != shim_col`, extend the forward/return routes with horizontal hops (mirror the `aie_control.cpp` EAST/WEST `StrmConnCctEnable` pattern). Keep the function < 200 lines.
- **Struct field access**: in aie_runtime.c use real `struct XAie_TxnCmd` fields; the offset-based reader lives only in the standalone test.
