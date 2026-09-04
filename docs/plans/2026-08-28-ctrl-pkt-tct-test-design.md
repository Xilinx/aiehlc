# Control-Packet TCT Test API — Design

**Date:** 2026-08-28
**Component:** AIE runtime (`src/mlir/runtime/aie_runtime.{c,h}`)
**Status:** Approved (design phase)

## Goal

Add a single runtime API that exercises the control-packet path end to end:

1. Accept a transaction buffer (raw `XAie_TxnCmd` stream).
2. Convert the transaction's register-write ops into control packets.
3. Set up the stream/packet-switch routing from a SHIM column to a destination
   tile's CTRL port (self-contained — the API programs the route).
4. Arm the TCT (task-completion-token) return path: the destination emits a
   token that routes back to a SHIM S2MM channel writing a token word into DDR.
5. Push the control packet, poll the DDR token for the done/apply TCT.
6. Measure host wall-time (push-start -> TCT-done) and throughput.

## Requirements Mapping

| User requirement | Design element |
|------------------|----------------|
| #1 single api | one public `__Runtime_ctrl_pkt_test`, internal static helpers |
| #2 transaction buffer as input | `const XAie_TxnCmd *txn, uint32_t txn_count` |
| #3 convert txn -> ctrl pkt, send to dest | `ctrl_txn_to_words` + reuse `__Runtime_ctrl_pktize` + `__Runtime_ctrl_push` |
| #4 routing logic for ctrl pkt to dest | `ctrl_route_setup` (StrmConnCct + PktSw Slave/Mstr, DONOT_DROP_HEADER) |
| #5 ctrl-pkt done/apply triggered TCT | return-path S2MM into DDR token buffer, polled |
| #6 routing includes the TCT | `ctrl_route_setup` also programs dest -> SHIM S2MM return route |
| #7 timing + throughput | host XTime around push+poll; `elapsed_us`, `throughput_mbps` |

## Approach

Approach B (chosen): **one public API + internal static helpers.** The public
function is a thin orchestrator; each stage is a `static` helper kept under the
project's 200-line-per-function limit. Reuses the existing control-packet
primitives (`__Runtime_ctrl_pktize`, `__Runtime_ctrl_push`) and buffer alloc
(`__Runtime_alloc_buffer`).

## Public API

```c
typedef struct {
    AieRC    rc;              // XAIE_OK on success
    uint32_t pkt_words;       // control-packet words actually sent
    uint32_t payload_bytes;   // register-write payload bytes (excl. headers)
    uint64_t elapsed_us;      // host wall time: push-start -> TCT-done
    double   throughput_mbps; // payload_bytes / elapsed_us
    uint32_t tct_value;       // token word observed in DDR
} __Runtime_CtrlPktTestResult;

__Runtime_CtrlPktTestResult
__Runtime_ctrl_pkt_test(XAie_DevInst *dev,
                        const struct XAie_TxnCmd *txn, uint32_t txn_count,
                        uint8_t shim_col, uint8_t dest_col, uint8_t dest_row,
                        uint8_t ctrl_stream_id, uint8_t mm2s_ch, uint8_t s2mm_ch);
```

## Internal Components (all `static`, each < 200 lines)

1. `ctrl_txn_to_words(txn, txn_count, &tile_addr, data_out, cap) -> nwords`
   - Walks the `XAie_TxnCmd` stream. For `XAIE_IO_WRITE` / `XAIE_IO_BLOCKWRITE`
     collects `RegOff` -> `Value`/`DataPtr` words into a flat reg-write buffer;
     records the base `tile_addr`. `XAIE_IO_CUSTOM_OP_TCT` and other opcodes are
     recorded as the completion/wait marker.
2. Reuse `__Runtime_ctrl_pktize(...)` to turn the reg words into control-packet
   words (two in-band headers per <=4-word chunk).
3. `ctrl_route_setup(dev, shim_col, dest_col, dest_row, stream_id, s2mm_ch)`
   - Forward route: `XAie_StrmConnCctEnable` hops SHIM -> dest, then
     `XAie_StrmPktSwSlaveSlotEnable` / `XAie_StrmPktSwMstrPortEnable` into the
     dest CTRL port with `XAIE_SS_PKT_DONOT_DROP_HEADER`.
   - Return route: dest -> SHIM S2MM for the TCT token.
4. `tct_s2mm_arm(dev, shim_col, s2mm_ch, token_buf)` — SHIM S2MM BD writing the
   TCT token into a DDR buffer (`__Runtime_alloc_buffer`).
5. Reuse `__Runtime_ctrl_push(...)` for the MM2S send.
6. `tct_poll(dev, token_buf, shim_col, s2mm_ch) -> value` — invalidate cache and
   poll the DDR token word, with S2MM pending-BD-count as a backstop, until the
   completion token appears.

Public `__Runtime_ctrl_pkt_test` sequences 1->6 and fills the result struct.

## Data Flow & Timing

```
XAie_TxnCmd[] --(1 parse)--> reg words + tile_addr
             --(2 pktize)--> ctrl-pkt words in DDR buf
route setup (3): SHIM(shim_col) MM2S --CTRL packet--> dest(col,row) CTRL port
return route:    dest --TCT token--> SHIM(shim_col) S2MM --> DDR token_buf
  t0 = XTime_GetTime()
  push (5) MM2S ; poll (6) token_buf for TCT
  t1 = XTime_GetTime()
elapsed_us = (t1-t0)/cps*1e6 ; throughput = payload_bytes/elapsed_us
```

Timing uses the runtime's existing host-clock source (XTime / `COUNTS_PER_SECOND`).
Under `__AIESIM__`, wall-time is meaningless: report `pkt_words`/`payload_bytes`,
set `elapsed_us=0`, and use `ess_WriteGM` for buffer staging like the existing
control path.

## Error Handling

- Every XAie call return-checked; first failure short-circuits, returned in
  `result.rc` with a `printf` diagnostic (matches existing runtime style).
- Packetize capacity overflow -> `rc=XAIE_ERR`.
- Buffers via `__Runtime_alloc_buffer`, freed before return.

## Testing

- Unit test `testCtrlPktTest()` in
  `src/mlir/mlirfront/tilinglinalg/pass/unitest/test.cpp`: build a small
  `XAie_TxnCmd[]` (2-3 WRITE ops + a TCT marker), call the parse/packetize
  helpers, assert control-packet word count/header layout match
  `__Runtime_ctrl_pktize` expectations (host-side logic test, no live device —
  consistent with existing `testControlPacketCtrlSink`).
- Full device path exercised via the normal HW-run flow (`apppaltest.py`).

## Files to Change

- `src/mlir/runtime/aie_runtime.h` — result struct + public prototype + doc.
- `src/mlir/runtime/aie_runtime.c` — public API + static helpers.
- `src/mlir/mlirfront/tilinglinalg/pass/unitest/test.cpp` — `testCtrlPktTest()`.
- `src/mlir/mlirfront/tilinglinalg/pass/unitest/CMakeLists.txt` — if new sources.
- `CLAUDE.md` / this doc — keep architecture docs current.
