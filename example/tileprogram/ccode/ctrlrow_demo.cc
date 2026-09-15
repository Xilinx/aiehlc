/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

/*
 * ctrlrow_demo.cc — row-based control connection demo (Task 9 multi-row E2E).
 *
 * Exercises the stateful row-control fabric API (design:
 * 2026-09-08-row-control-connection):
 *   __Runtime_ctrl_plan_init - init the fabric on a shim column and configure
 *                              every EAST chain in one call (each packet climbs a
 *                              shared vertical spine to the row's left tile, then
 *                              daisy-chains EAST; the static planner marks the top
 *                              row so its return head omits the idle RET_NORTH slot)
 *   __Runtime_ctrl_row_broadcast_write - fire a WRITE control packet (id[4]=1)
 *                              that every tile on every configured row consumes
 *   __Runtime_ctrl_row_whole_row_write - fire a WRITE control packet
 *                              (id = (ACR_CLASS_WHOLE_ROW<<2)|rowidx) that every
 *                              column of one target row consumes. The packet climbs
 *                              the shared spine past any intervening head.
 *   __Runtime_ctrl_row_read       - read a tile addr from every column of a row via
 *                              the merged westbound return chain (one response
 *                              packet per column, kept header routes it to its col)
 *   __Runtime_ctrl_row_write_ack  - whole-row write-with-ack; every column returns a
 *                              header-only ack that drains the shim S2MM (barrier)
 *   __Runtime_ctrl_row_close - tear down
 *
 * The E2E flow configures TWO core rows on one shared spine to prove spine reuse
 * and the forward + return control fabric:
 *   1-3. plan_init kRows[]={A(lower,idx0), B(higher,idx1)} -> in one call the spine
 *        climbs shim..A then EXTENDS to B reusing hops shim..A; B is the top row so
 *        its return head omits the idle RET_NORTH slot.
 *   4. broadcast_write sentinel to a scratch L1 addr on every tile of both rows
 *   5. whole-row write to row A then row B, verifying via the direct debug memory
 *      interface that every column of the targeted row changed and the other row
 *      did not (row B is ABOVE head row A, so this also checks the spine climb past
 *      a head).
 *   6. read every column of both rows through the return chain and compare each
 *      column's returned word to a per-column seed (exercises the merged westbound
 *      return path + per-column src_col routing).
 *   7. whole-row write-with-ack on both rows; every column's ack must drain.
 * Success = every probe PASSes and the run reaches teardown with no AIE ERROR.
 *
 * Build (single-kernel flow — the dummy kernel below is extracted into a kernel
 * ELF; aie_runtime.c + aie_runtime_control_plan.c are linked in):
 *   source script/aiehlc.sh --aie-version 5 \
 *       --runtime-source-file example/tileprogram/ccode/ctrlrow_demo.cc
 * Run:
 *   python3 script/test/apppaltest.py -y -nonreboot > ./applog 2>&1
 */

#include "xaiengine.h"
#include <stdio.h>

#ifndef __AIESIM__
#include "xil_cache.h"
#include "xil_io.h"
#include "xil_printf.h"
#include "sleep.h"
#else
#include <unistd.h>
#endif /* __AIESIM__ */

#if AIE_GEN <= 2
#define HW_GEN XAIE_DEV_GEN_AIEML
#else
#define HW_GEN XAIE_DEV_GEN_AIE2PS
#endif

#if AIE_GEN <= 2
#define XAIE_BASE_ADDR 0x20000000000
#define XAIE_COL_SHIFT 25
#define XAIE_ROW_SHIFT 20
#define XAIE_NUM_ROWS 11
#define XAIE_NUM_COLS 38
#define XAIE_SHIM_ROW 0
#define XAIE_RES_TILE_ROW_START 1
#define XAIE_RES_TILE_NUM_ROWS 2
#define XAIE_AIE_TILE_ROW_START 3
#define XAIE_AIE_TILE_NUM_ROWS 8
#else
#define XAIE_BASE_ADDR 0x20000000000
#define XAIE_COL_SHIFT 25
#define XAIE_ROW_SHIFT 20
#define XAIE_NUM_ROWS 7
#define XAIE_NUM_COLS 36
#define XAIE_SHIM_ROW 0
#define XAIE_RES_TILE_ROW_START 1
#define XAIE_RES_TILE_NUM_ROWS 2
#define XAIE_AIE_TILE_ROW_START 3
#define XAIE_AIE_TILE_NUM_ROWS 4
#endif

// __Runtime_* wrappers + row-control API, linked in from src/mlir/runtime by
// aiehlc.sh's RUNTIME_SRCS. Included after xaiengine.h.
#include "aie_runtime.h"

// Dummy kernel so the aiehlc single-kernel flow still produces a kernel ELF. The
// cores never compute in this control-plane demo; the body is intentionally
// empty.
__global__ void ctrlrow_demo_dummy(input_window_int32 *win
                                   __attribute__((annotate("mem_address:0x1000"), annotate("size_hint:512"))),
                                   output_window_int32 *out
                                   __attribute__((annotate("mem_address:0x6000"), annotate("size_hint:512")))) {
    (void)win;
    (void)out;
}

// Demo geometry (AIE2PS): shim row 0, memtile rows 1-2, core rows 3-6.
#define DEMO_SHIM_COL 0u
#define DEMO_CTRL_ID 0u
#define DEMO_CORE_ROW_A 3u /* lower core row (spine climbs shim..A first) */
#define DEMO_CORE_ROW_B 5u /* higher core row (spine extends A..B, reuses shim..A) */
#define DEMO_COL_LO 0u
#define DEMO_COL_HI 3u            /* 2-tile chain: head consumes+forwards, east consumes */
#define DEMO_SCRATCH_ADDR 0x1000u /* in-range core L1 (DMA-view) scratch */
#define DEMO_SENTINEL 0xC0DE0001u
#define DEMO_BD_ID 2
#define DEMO_MM2S_CH 0
#define DEMO_S2MM_CH 0 /* shim response S2MM (even ch -> bd<24, parity rule) */

/* Column-subset multicast: a distinct value per probe so both cross-column and
 * cross-row leaks are visible. Verified via the direct debug memory interface
 * (XAie_DataMemBlockRead), not a control-response readback. */
#define DEMO_MC_ADDR 0x2000u /* separate in-range core L1 scratch */
#define DEMO_MC_BD 3         /* shim MM2S BD for the multicast send */

/* Read / write-ack demo scratch. The return path funnels one response packet per
 * column down the spine to the shim S2MM: the read/ack call uses forward MM2S BD
 * DEMO_RW_BD and return S2MM BDs DEMO_RW_BD+1 .. DEMO_RW_BD+ncols (ncols=4 here),
 * so DEMO_RW_BD is chosen to leave room for 4 sequential return BDs (<16). */
#define DEMO_RD_ADDR 0x3000u   /* read-back scratch (seeded per column) */
#define DEMO_RD_BASE 0xBEEF00u /* per-column seed = DEMO_RD_BASE | col */
#define DEMO_ACK_ADDR 0x4000u  /* write-with-ack scratch */
#define DEMO_RW_BD 8           /* fwd BD; return BDs 9..12 (ncols=4) */

/* Whole-row multicast @val to every column of @target_row at DEMO_MC_ADDR, then
 * verify via the direct debug memory interface that EVERY column of @target_row
 * holds @val and no column of the other configured row @other_row does (no
 * cross-row leak). Distinct @val per call keeps stale writes out of the equality
 * test. Returns 0 on PASS, -1 otherwise. (The forward chain now carries only the
 * uniform CONSUME + BCAST classes; whole-row is the single column-subset probe.) */
static int demo_probe_whole_row(__Runtime_CtrlRowFabric *fab, XAie_DevInst *dev, uint8_t target_row, uint8_t other_row,
                                uint32_t val) {
    uint32_t wr = val;
    AieRC rc = __Runtime_ctrl_row_whole_row_write(fab, target_row, DEMO_MC_ADDR, &wr, 1u, DEMO_MC_BD, DEMO_MM2S_CH, 1);
    if (rc != XAIE_OK) {
        printf("[ctrlrow] whole-row write row=%u rc=%d\n", (unsigned)target_row, (int)rc);
        return -1;
    }
    printf("[ctrlrow] whole-row write 0x%08x -> addr 0x%x on row %u done\n", val, DEMO_MC_ADDR, (unsigned)target_row);

    int fails = 0;
    /* Target row: every column must hold @val. */
    for (uint8_t dc = DEMO_COL_LO; dc <= DEMO_COL_HI; dc++) {
        uint32_t v = 0u;
        (void)XAie_DataMemBlockRead(dev, XAie_TileLoc(dc, target_row), DEMO_MC_ADDR, &v, sizeof(v));
        int ok = (v == val);
        printf("[ctrlrow] whole-row row=%u tile(%u,%u) -> 0x%08x %s (expect SET) %s\n", (unsigned)target_row,
               (unsigned)dc, (unsigned)target_row, v, ok ? "SET" : "unset", ok ? "PASS" : "FAIL");
        if (!ok)
            fails++;
    }
    /* Other row: no column may hold this value (no cross-row leak). */
    for (uint8_t dc = DEMO_COL_LO; dc <= DEMO_COL_HI; dc++) {
        uint32_t v = 0u;
        (void)XAie_DataMemBlockRead(dev, XAie_TileLoc(dc, other_row), DEMO_MC_ADDR, &v, sizeof(v));
        int ok = (v != val);
        printf("[ctrlrow] whole-row row=%u no-leak tile(%u,%u) -> 0x%08x %s\n", (unsigned)target_row, (unsigned)dc,
               (unsigned)other_row, v, ok ? "PASS" : "LEAK");
        if (!ok)
            fails++;
    }
    return fails == 0 ? 0 : -1;
}

/* Read demo: seed a DISTINCT value into DEMO_RD_ADDR on every column of @row via
 * the direct debug memory interface, then read the whole row back through the
 * control-packet return path (__Runtime_ctrl_row_read) and confirm each column's
 * returned word matches its seed. This exercises the merged westbound return chain
 * and per-column src_col routing (each column's response keeps its stream header).
 * Returns 0 on PASS, -1 otherwise. */
static int demo_probe_read(__Runtime_CtrlRowFabric *fab, XAie_DevInst *dev, uint8_t row) {
    int ncols = (int)(DEMO_COL_HI - DEMO_COL_LO + 1u);
    for (uint8_t dc = DEMO_COL_LO; dc <= DEMO_COL_HI; dc++) {
        uint32_t seed = DEMO_RD_BASE | (uint32_t)dc;
        (void)XAie_DataMemBlockWrite(dev, XAie_TileLoc(dc, row), DEMO_RD_ADDR, &seed, sizeof(seed));
    }
    uint32_t out_vals[DEMO_COL_HI - DEMO_COL_LO + 1u];
    for (int i = 0; i < ncols; i++)
        out_vals[i] = 0u;
    AieRC rc = __Runtime_ctrl_row_read(fab, row, DEMO_RD_ADDR, 1u, out_vals, DEMO_RW_BD, DEMO_MM2S_CH);
    if (rc != XAIE_OK) {
        printf("[ctrlrow] row_read row=%u rc=%d\n", (unsigned)row, (int)rc);
        return -1;
    }
    int fails = 0;
    for (uint8_t dc = DEMO_COL_LO; dc <= DEMO_COL_HI; dc++) {
        int ci = (int)(dc - DEMO_COL_LO);
        uint32_t expect = DEMO_RD_BASE | (uint32_t)dc;
        int ok = (out_vals[ci] == expect);
        printf("[ctrlrow] read row=%u col=%u -> 0x%08x (expect 0x%08x) %s\n", (unsigned)row, (unsigned)dc, out_vals[ci],
               expect, ok ? "PASS" : "FAIL");
        if (!ok)
            fails++;
    }
    return fails == 0 ? 0 : -1;
}

/* Write-ack demo: whole-row write-with-ack @val to DEMO_ACK_ADDR across @row; every
 * column returns a header-only ack that drains the shim S2MM. A XAIE_OK return means
 * all ncols acks drained (the completion barrier). Then confirm the value landed on
 * every column via the direct debug interface. Returns 0 on PASS, -1 otherwise. */
static int demo_probe_write_ack(__Runtime_CtrlRowFabric *fab, XAie_DevInst *dev, uint8_t row, uint32_t val) {
    uint32_t wr = val;
    AieRC rc = __Runtime_ctrl_row_write_ack(fab, row, DEMO_ACK_ADDR, &wr, 1u, DEMO_RW_BD, DEMO_MM2S_CH);
    if (rc != XAIE_OK) {
        printf("[ctrlrow] row_write_ack row=%u rc=%d (not all acks drained)\n", (unsigned)row, (int)rc);
        return -1;
    }
    printf("[ctrlrow] write-ack 0x%08x -> addr 0x%x on row %u: all acks drained PASS\n", val, DEMO_ACK_ADDR,
           (unsigned)row);
    int fails = 0;
    for (uint8_t dc = DEMO_COL_LO; dc <= DEMO_COL_HI; dc++) {
        uint32_t v = 0u;
        (void)XAie_DataMemBlockRead(dev, XAie_TileLoc(dc, row), DEMO_ACK_ADDR, &v, sizeof(v));
        int ok = (v == val);
        printf("[ctrlrow] write-ack row=%u tile(%u,%u) -> 0x%08x %s\n", (unsigned)row, (unsigned)dc, (unsigned)row, v,
               ok ? "PASS" : "FAIL");
        if (!ok)
            fails++;
    }
    return fails == 0 ? 0 : -1;
}

int run_ctrlrow_demo(XAie_DevInst *dev) {
    __Runtime_CtrlRowFabric fab;

    /* Emit the control-plan provenance map (CONTROLPAN-PMAP lines) so the
     * aiedebug device-map "Load control plan" button can overlay the row-fabric
     * control route. Must precede plan_init. On this baremetal target
     * getenv("AIE_CTRL_PMAP") is unavailable, so enable it explicitly here. */
    __Runtime_ctrl_pmap_enable(1);

    /* 1-3. One-shot init: plan + emit both EAST chains on the shared spine. Row A
     * (lower) climbs shim..A; row B (higher) EXTENDS the spine to B, reusing the
     * shim..A hops. The static planner marks the top row so its return head omits
     * the idle RET_NORTH slot. Rows are listed bottom-up. */
    static const __Runtime_CtrlRowChain kRows[] = {
        {DEMO_CORE_ROW_A, DEMO_COL_LO, DEMO_COL_HI},
        {DEMO_CORE_ROW_B, DEMO_COL_LO, DEMO_COL_HI},
    };
    AieRC rc = __Runtime_ctrl_plan_init(&fab, dev, DEMO_SHIM_COL, DEMO_S2MM_CH, (uint8_t)DEMO_CTRL_ID, kRows,
                                        (uint8_t)(sizeof(kRows) / sizeof(kRows[0])));
    if (rc != XAIE_OK) {
        printf("[ctrlrow] plan_init rc=%d\n", (int)rc);
        return -1;
    }
    printf("[ctrlrow] plan_init rows=%u spine_top=%u (A=%u,B=%u share spine)\n", (unsigned)fab.nrows,
           (unsigned)fab.spine.spine_top, DEMO_CORE_ROW_A, DEMO_CORE_ROW_B);

    /* 4. Broadcast a sentinel to every tile of both configured rows. */
    uint32_t sentinel = DEMO_SENTINEL;
    rc = __Runtime_ctrl_row_broadcast_write(&fab, DEMO_SCRATCH_ADDR, &sentinel, /*nwords=*/1u, DEMO_BD_ID, DEMO_MM2S_CH,
                                            /*log=*/1);
    if (rc != XAIE_OK) {
        printf("[ctrlrow] broadcast_write rc=%d\n", (int)rc);
        __Runtime_ctrl_row_close(&fab);
        return -1;
    }
    printf("[ctrlrow] broadcast_write 0x%08x -> addr 0x%x on rows %u,%u done\n", DEMO_SENTINEL, DEMO_SCRATCH_ADDR,
           DEMO_CORE_ROW_A, DEMO_CORE_ROW_B);

    /* DIAG: direct-read the broadcast sentinel at each row's head+target to see how
     * far the spine climb (msel0) actually reaches (isolates NORTH-up fan-out). */
    for (uint8_t dr = DEMO_CORE_ROW_A; dr <= DEMO_CORE_ROW_B; dr += (uint8_t)(DEMO_CORE_ROW_B - DEMO_CORE_ROW_A)) {
        for (uint8_t dc = DEMO_COL_LO; dc <= DEMO_COL_HI; dc++) {
            uint32_t v = 0u;
            (void)XAie_DataMemBlockRead(dev, XAie_TileLoc(dc, dr), DEMO_SCRATCH_ADDR, &v, sizeof(v));
            printf("[ctrlrow] DIAG bcast tile(%u,%u) addr 0x%x -> 0x%08x\n", (unsigned)dc, (unsigned)dr,
                   DEMO_SCRATCH_ADDR, v);
        }
    }

    /* 5. Whole-row column-subset writes, verified via the direct debug memory
     * interface. Exercise whole-row on row A (lower head) and on row B (ABOVE head
     * A -> also checks the spine climb past a head). Each probe checks every column
     * of the target row changed and the other row did not (no cross-row leak).
     * Distinct values keep the checks clean. */
    int fails = 0;
    fails += demo_probe_whole_row(&fab, dev, DEMO_CORE_ROW_A, DEMO_CORE_ROW_B, 0x00110003u) ? 1 : 0;
    fails += demo_probe_whole_row(&fab, dev, DEMO_CORE_ROW_B, DEMO_CORE_ROW_A, 0x00110005u) ? 1 : 0;

    /* 6. Read demo: read every column of both rows back through the merged
     * westbound return chain (each column returns its own response packet, kept
     * header routes it to its column). */
    fails += demo_probe_read(&fab, dev, DEMO_CORE_ROW_A) ? 1 : 0;
    fails += demo_probe_read(&fab, dev, DEMO_CORE_ROW_B) ? 1 : 0;

    /* 7. Write-ack demo: whole-row write-with-ack on both rows; every column's ack
     * must drain the shim S2MM (the completion barrier). */
    fails += demo_probe_write_ack(&fab, dev, DEMO_CORE_ROW_A, 0xACCA0003u) ? 1 : 0;
    fails += demo_probe_write_ack(&fab, dev, DEMO_CORE_ROW_B, 0xACCA0005u) ? 1 : 0;

    __Runtime_ctrl_row_close(&fab);
    printf("[ctrlrow] DONE %s (%d multicast failure(s))\n", fails == 0 ? "PASS" : "FAIL", fails);
    return fails == 0 ? 0 : -1;
}

/* Transaction-capture row-multicast scratch. Each per-row transaction captures
 * one BlockWrite32 into core L1, commits it as a row-multicast (commit row =
 * target physical row), proves the committed control-packet buffer equals a
 * direct pktize with the row-multicast stream id, then delivers via the fabric
 * whole-row primitive and verifies every column of the row landed the payload. */
#define DEMO_TXN_MC_ADDR 0x5000u /* BlockWrite32 target (in-range core L1 scratch) */

/* 4x4 broadcast transaction scratch. demo_txn_multipl_row captures ONE
 * BlockWrite32 of DEMO_TXN_MR_NW words at DEMO_TXN_MR_ADDR and commits it with
 * the tile mapped to the broadcast stream id (ACR_ID_BCAST, id[4]=1). Because a
 * single <=4-word access packetizes identically whether it comes from the commit
 * translator or a direct pktize, the committed buffer is byte-identical to the
 * words __Runtime_ctrl_row_broadcast_write emits — verified before the same
 * payload is broadcast to every tile of a 4-row x 4-col grid. */
#define DEMO_TXN_MR_ADDR 0x6000u   /* 16-byte-aligned core L1 (one 128b access) */
#define DEMO_TXN_MR_NW 4u          /* 4 words = exactly one control-packet access */
#define DEMO_TXN_MR_BASE 0x4B0000u /* payload word i = DEMO_TXN_MR_BASE | i */
#define DEMO_TXN_MR_BD 6           /* shim MM2S BD for the broadcast send */

/* Capture one BlockWrite32 as a control-packet transaction and MULTICAST it to
 * every column of @row via the row-control fabric @fab. Flow:
 *   1. start_transaction (nothing touches HW; DISABLE_AUTO_FLUSH)
 *   2. XAie_BlockWrite32 of DEMO_TXN_MR_NW words at DEMO_TXN_MC_ADDR targeting a
 *      core tile ON @row (the commit validates every captured op lands on @row)
 *   3. commit with row=@row -> the translator stamps the row-multicast stream id
 *      (ACR_CLASS_WHOLE_ROW<<2)|rowidx (id[4]=0) on the packet
 *   4. prove the committed buffer equals a direct pktize with that same sid
 *   5. deliver via __Runtime_ctrl_row_whole_row_write and verify every column of
 *      @row landed the payload (direct debug memory interface)
 * @bw_base seeds the payload (word i = @bw_base | i). Returns 0 on PASS. */
static int demo_txn_multicast_row(XAie_DevInst *dev, __Runtime_CtrlRowFabric *fab, uint8_t row, uint32_t bw_base) {
    uint32_t data[DEMO_TXN_MR_NW];
    for (uint32_t i = 0u; i < DEMO_TXN_MR_NW; i++)
        data[i] = bw_base | i;

    /* start_transaction records the cast target (row) into @fab; commit reads it
     * back (stateless w.r.t. the caller). */
    if (__Runtime_control_start_transaction(dev, fab, (int)row) != 0) {
        printf("[ctrltxn] start_transaction failed row=%u\n", (unsigned)row);
        return -1;
    }
    /* Every captured op must target @row (commit rejects otherwise). Use the
     * spine column of @row; the row-multicast reaches every column regardless. */
    uint64_t tile_base = ((uint64_t)DEMO_SHIM_COL << XAIE_COL_SHIFT) | ((uint64_t)row << XAIE_ROW_SHIFT);
    (void)XAie_BlockWrite32(dev, tile_base + DEMO_TXN_MC_ADDR, data, DEMO_TXN_MR_NW);

    uint32_t pkt[16] = {0u};
    uint32_t nwords = 0u;
    int rc = __Runtime_control_write_pkt_commit_transaction(dev, fab, pkt, (uint32_t)(sizeof(pkt) / sizeof(pkt[0])),
                                                            &nwords);
    if (rc != 0 || nwords == 0u) {
        printf("[ctrltxn] commit_transaction row=%u rc=%d nwords=%u\n", (unsigned)row, rc, (unsigned)nwords);
        return -1;
    }

    /* Prove the committed buffer IS the row-multicast packet: byte-match a direct
     * pktize with the same (ACR_CLASS_WHOLE_ROW<<2)|rowidx stream id. */
    uint8_t rowidx = 0u;
    for (uint8_t i = 0u; i < fab->nrows; i++)
        if (fab->rows[i].row == row) {
            rowidx = i;
            break;
        }
    uint8_t sid = (uint8_t)(((uint8_t)ACR_CLASS_WHOLE_ROW << 2) | (rowidx & 0x3u));
    uint32_t ref_pkt[16] = {0u};
    uint32_t ref_nw = __Runtime_ctrl_pktize_write(ref_pkt, (uint32_t)(sizeof(ref_pkt) / sizeof(ref_pkt[0])), sid,
                                                  DEMO_TXN_MC_ADDR, data, DEMO_TXN_MR_NW,
                                                  /*lastwriteack=*/0, /*ret_stream_id=*/0u, NULL);
    int match = (ref_nw == nwords) && (memcmp(pkt, ref_pkt, (size_t)nwords * sizeof(uint32_t)) == 0);
    printf("[ctrltxn] row=%u multicast commit -> %u words (sid=%u), ref=%u words: %s\n", (unsigned)row,
           (unsigned)nwords, (unsigned)sid, (unsigned)ref_nw, match ? "MATCH" : "MISMATCH");
    int fails = match ? 0 : 1;

    /* Issue the COMMITTED buffer itself via __Runtime_control_push: it reads the
     * cast target back from @fab (set by start_transaction), copies @pkt into a
     * device DMA buffer, and pushes it down the shared spine. This is the
     * capture->commit->push flow (no re-pktize), then verify every column of @row. */
    rc = __Runtime_control_push(fab, pkt, nwords, DEMO_TXN_MR_BD, DEMO_MM2S_CH, /*block=*/1, /*log=*/1);
    if (rc != XAIE_OK) {
        printf("[ctrltxn] control_push row=%u rc=%d\n", (unsigned)row, (int)rc);
        return -1;
    }
    for (uint8_t dc = DEMO_COL_LO; dc <= DEMO_COL_HI; dc++) {
        int tile_ok = 1;
        for (uint32_t i = 0u; i < DEMO_TXN_MR_NW; i++) {
            uint32_t v = 0u;
            (void)XAie_DataMemBlockRead(dev, XAie_TileLoc(dc, row), DEMO_TXN_MC_ADDR + i * 4u, &v, sizeof(v));
            if (v != data[i])
                tile_ok = 0;
        }
        printf("[ctrltxn] row=%u tile(%u,%u) @0x%x[0..%u] %s\n", (unsigned)row, (unsigned)dc, (unsigned)row,
               DEMO_TXN_MC_ADDR, (unsigned)(DEMO_TXN_MR_NW - 1u), tile_ok ? "PASS" : "FAIL");
        if (!tile_ok)
            fails++;
    }
    return fails == 0 ? 0 : -1;
}

/* run_ctrol_trasaction — control-plan transaction MULTICAST demo.
 *
 * Shows the capture->commit->push flow of the transaction API for row-cast:
 *   __Runtime_control_start_transaction(dev,fab,row)  begin capture, record cast
 *   XAie_BlockWrite32                                 record ops (no HW touch)
 *   __Runtime_control_write_pkt_commit_transaction(dev,fab,...) -> row-multicast words
 *   __Runtime_control_push(fab,pkt,nwords,...)        push committed words to HW
 *
 * Configures a fabric with two core rows (A, B) on the shared spine, then
 * multicasts a distinct payload to each row and verifies every column landed it
 * (cross-row leaks would show as a mismatch). Returns 0 on PASS, -1 otherwise. */
int run_ctrol_trasaction(XAie_DevInst *dev) {
    __Runtime_CtrlRowFabric fab;
    static const __Runtime_CtrlRowChain kRows[] = {
        {DEMO_CORE_ROW_A, DEMO_COL_LO, DEMO_COL_HI},
        {DEMO_CORE_ROW_B, DEMO_COL_LO, DEMO_COL_HI},
    };
    const uint8_t nrows = (uint8_t)(sizeof(kRows) / sizeof(kRows[0]));
    AieRC rc = __Runtime_ctrl_plan_init(&fab, dev, DEMO_SHIM_COL, DEMO_S2MM_CH, (uint8_t)DEMO_CTRL_ID, kRows, nrows);
    if (rc != XAIE_OK) {
        printf("[ctrltxn] plan_init rc=%d\n", (int)rc);
        return -1;
    }
    int fails = 0;
    fails += demo_txn_multicast_row(dev, &fab, (uint8_t)DEMO_CORE_ROW_A, /*bw_base=*/0x51A00000u) ? 1 : 0;
    fails += demo_txn_multicast_row(dev, &fab, (uint8_t)DEMO_CORE_ROW_B, /*bw_base=*/0x51B00000u) ? 1 : 0;
    __Runtime_ctrl_row_close(&fab);
    printf("[ctrltxn] DONE %s (%d transaction failure(s))\n", fails == 0 ? "PASS" : "FAIL", fails);
    return fails == 0 ? 0 : -1;
}

/* demo_txn_multipl_row — capture a control-plane write via the transaction API
 * and BROADCAST it to a 4x4 tile grid (4 rows x 4 cols = 16 tiles).
 *
 * demo_txn_one_tile pushes via __Runtime_ctrl_push_target, which is same-column
 * only (dest_col == shim_col), so it cannot reach a grid that spans 4 columns.
 * A 4x4 broadcast instead uses the row-control fabric broadcast primitive
 * (__Runtime_ctrl_row_broadcast_write): one stream-id-ACR_ID_BCAST packet climbs
 * the shared spine and every tile of every configured row consumes it.
 *
 * The transaction API is still the STAR: a single BlockWrite32 committed with the
 * (representative) tile mapped to ACR_ID_BCAST yields exactly the broadcast packet
 * words. Since a <=4-word access packetizes identically from the commit translator
 * and from a direct pktize, the committed buffer is byte-compared to a reference
 * pktize to prove the equivalence; the same payload is then broadcast to the grid
 * and all 16 tiles are verified via the direct debug memory interface.
 * Returns 0 on PASS, -1 otherwise. */
int demo_txn_multipl_row(XAie_DevInst *dev) {
    __Runtime_CtrlRowFabric fab;

    /* Emit the CONTROLPAN-PMAP provenance lines (aiedebug overlay). Must precede
     * plan_init; getenv("AIE_CTRL_PMAP") is unavailable on this baremetal target. */
    __Runtime_ctrl_pmap_enable(1);

    /* Configure a 4x4 grid: 4 core rows, each spanning cols DEMO_COL_LO..DEMO_COL_HI,
     * all rooted on the shared spine at DEMO_SHIM_COL. Rows are listed bottom-up;
     * the static planner marks the top row so its return head omits the idle
     * RET_NORTH slot. 4 rows == the row-multicast index limit (id[1:0]), but a
     * broadcast (id[4]=1) reaches every configured row regardless. */
    static const __Runtime_CtrlRowChain kRows[] = {
        {3u, DEMO_COL_LO, DEMO_COL_HI},
        {4u, DEMO_COL_LO, DEMO_COL_HI},
        {5u, DEMO_COL_LO, DEMO_COL_HI},
        {6u, DEMO_COL_LO, DEMO_COL_HI},
    };
    const uint8_t nrows = (uint8_t)(sizeof(kRows) / sizeof(kRows[0]));
    AieRC rc = __Runtime_ctrl_plan_init(&fab, dev, DEMO_SHIM_COL, DEMO_S2MM_CH, (uint8_t)DEMO_CTRL_ID, kRows, nrows);
    if (rc != XAIE_OK) {
        printf("[ctrltxn-mr] plan_init rc=%d\n", (int)rc);
        return -1;
    }
    printf("[ctrltxn-mr] plan_init rows=%u cols=%u..%u (4x4 grid on spine col %u)\n", (unsigned)fab.nrows,
           (unsigned)DEMO_COL_LO, (unsigned)DEMO_COL_HI, (unsigned)DEMO_SHIM_COL);

    /* 1. Payload: DEMO_TXN_MR_NW words, word i = base | i. */
    uint32_t data[DEMO_TXN_MR_NW];
    for (uint32_t i = 0u; i < DEMO_TXN_MR_NW; i++)
        data[i] = DEMO_TXN_MR_BASE | i;

    /* 2. Capture the write in a transaction and commit it as a whole-array
     * BROADCAST (row = -1). The commit validates every captured op targets a CORE
     * tile (broadcast reaches cores) and stamps ACR_ID_BCAST (id[4]=1) on the
     * packet. The representative tile only supplies the RegOff (col/row/addr): use
     * a core tile on the spine head so the core-tile check passes. */
    const uint8_t rep_col = (uint8_t)DEMO_SHIM_COL, rep_row = kRows[0].row;
    /* start_transaction records the broadcast cast target (row = -1) into @fab;
     * commit reads it back (stateless w.r.t. the caller). */
    if (__Runtime_control_start_transaction(dev, &fab, /*row=*/-1) != 0) {
        printf("[ctrltxn-mr] start_transaction failed\n");
        __Runtime_ctrl_row_close(&fab);
        return -1;
    }
    uint64_t tile_base = ((uint64_t)rep_col << XAIE_COL_SHIFT) | ((uint64_t)rep_row << XAIE_ROW_SHIFT);
    (void)XAie_BlockWrite32(dev, tile_base + DEMO_TXN_MR_ADDR, data, DEMO_TXN_MR_NW);

    uint32_t txn_pkt[16] = {0u};
    uint32_t txn_nw = 0u;
    int trc = __Runtime_control_write_pkt_commit_transaction(dev, &fab, txn_pkt,
                                                             (uint32_t)(sizeof(txn_pkt) / sizeof(txn_pkt[0])), &txn_nw);
    if (trc != 0 || txn_nw == 0u) {
        printf("[ctrltxn-mr] commit_transaction rc=%d nwords=%u\n", trc, (unsigned)txn_nw);
        __Runtime_ctrl_row_close(&fab);
        return -1;
    }

    /* 3. Prove the committed buffer IS the broadcast packet: byte-match it to a
     * direct pktize of the same payload with stream id ACR_ID_BCAST (exactly what
     * __Runtime_ctrl_row_broadcast_write emits internally). */
    uint32_t ref_pkt[16] = {0u};
    uint32_t ref_nw = __Runtime_ctrl_pktize_write(ref_pkt, (uint32_t)(sizeof(ref_pkt) / sizeof(ref_pkt[0])),
                                                  (uint32_t)ACR_ID_BCAST, DEMO_TXN_MR_ADDR, data, DEMO_TXN_MR_NW,
                                                  /*lastwriteack=*/0, /*ret_stream_id=*/0u, NULL);
    int match = (ref_nw == txn_nw) && (memcmp(txn_pkt, ref_pkt, (size_t)txn_nw * sizeof(uint32_t)) == 0);
    uint32_t pid = 0u, ptype = 0u, prow = 0u, pcol = 0u;
    (void)__Runtime_ctrl_parse_pkt_hdr(txn_pkt[0], &pid, &ptype, &prow, &pcol);
    printf("[ctrltxn-mr] txn commit -> %u words, pkt id=%u (bcast id=%u), ref=%u words: %s\n", (unsigned)txn_nw,
           (unsigned)pid, (unsigned)ACR_ID_BCAST, (unsigned)ref_nw, match ? "MATCH" : "MISMATCH");
    int fails = match ? 0 : 1;

    /* 4. Issue the COMMITTED broadcast buffer itself via __Runtime_control_push:
     * it reads the cast target back from @fab (row = -1 => ACR_ID_BCAST), copies
     * @txn_pkt into a device DMA buffer, arms the shim spine entry, and fires the
     * packet to every tile of the 4x4 grid. This is the capture->commit->push flow
     * (no re-pktize inside the broadcast primitive). */
    rc = __Runtime_control_push(&fab, txn_pkt, txn_nw, DEMO_TXN_MR_BD, DEMO_MM2S_CH, /*block=*/1, /*log=*/1);
    if (rc != XAIE_OK) {
        printf("[ctrltxn-mr] control_push rc=%d\n", (int)rc);
        __Runtime_ctrl_row_close(&fab);
        return -1;
    }
    printf("[ctrltxn-mr] broadcast 0x%08x.. -> addr 0x%x on rows 3..6 done\n", data[0], DEMO_TXN_MR_ADDR);

    /* 5. Verify all 16 tiles hold the DEMO_TXN_MR_NW words. */
    for (uint8_t ri = 0u; ri < nrows; ri++) {
        uint8_t dr = kRows[ri].row;
        for (uint8_t dc = DEMO_COL_LO; dc <= DEMO_COL_HI; dc++) {
            int tile_ok = 1;
            for (uint32_t i = 0u; i < DEMO_TXN_MR_NW; i++) {
                uint32_t v = 0u;
                (void)XAie_DataMemBlockRead(dev, XAie_TileLoc(dc, dr), DEMO_TXN_MR_ADDR + i * 4u, &v, sizeof(v));
                if (v != data[i])
                    tile_ok = 0;
            }
            printf("[ctrltxn-mr] tile(%u,%u) @0x%x[0..%u] %s\n", (unsigned)dc, (unsigned)dr, DEMO_TXN_MR_ADDR,
                   (unsigned)(DEMO_TXN_MR_NW - 1u), tile_ok ? "PASS" : "FAIL");
            if (!tile_ok)
                fails++;
        }
    }

    __Runtime_ctrl_row_close(&fab);
    printf("[ctrltxn-mr] DONE %s (%d failure(s))\n", fails == 0 ? "PASS" : "FAIL", fails);
    return fails == 0 ? 0 : -1;
}

int main(int argc, char *argv[]) {
    (void)argc;
    (void)argv;

    XAie_SetupConfig(ConfigPtr, HW_GEN, XAIE_BASE_ADDR, XAIE_COL_SHIFT, XAIE_ROW_SHIFT, XAIE_NUM_COLS, XAIE_NUM_ROWS,
                     XAIE_SHIM_ROW, XAIE_RES_TILE_ROW_START, XAIE_RES_TILE_NUM_ROWS, XAIE_AIE_TILE_ROW_START,
                     XAIE_AIE_TILE_NUM_ROWS);

    XAie_InstDeclare(DevInst, &ConfigPtr);

    AieRC RC = XAie_CfgInitialize(&DevInst, &ConfigPtr);
    if (RC != XAIE_OK) {
        printf("Driver initialization failed.\n");
        return -1;
    }

#ifdef __AIESIM__
    XAie_SetIOBackend(&DevInst, XAIE_IO_BACKEND_SIM);
#else
    XAie_SetIOBackend(&DevInst, XAIE_IO_BACKEND_BAREMETAL);
#if AIE_GEN >= 2
    if (DevInst.Backend->Type == XAIE_IO_BACKEND_BAREMETAL) {
#if AIE_GEN == 5
        RC = XAie_UpdateNpiAddr(&DevInst, 0xf6d50000);
#else
        RC = XAie_UpdateNpiAddr(&DevInst, 0xF6D10000);
#endif
        if (RC != XAIE_OK) {
            printf("Failed to update NPI address\n");
            return -1;
        }
    }
    XAie_PartInitOpts PartInitOpts;
    PartInitOpts.Locs = NULL;
    PartInitOpts.NumUseTiles = 0;
    PartInitOpts.InitOpts = XAIE_PART_INIT_OPT_DEFAULT | XAIE_PART_INIT_OPT_CTRL_TLASTERROR_DISABLE;
    RC = XAie_PartitionInitialize(&DevInst, &PartInitOpts);
#else
    XAie_PmRequestTiles(&DevInst, NULL, 0);
#endif
#endif /* __AIESIM__ */

    // run_ctrlrow_demo(&DevInst);
    /* Control-plan transaction demo: capture write-only ops into an XAie
     * transaction, commit them into control-packet words, and push them. */
    // run_ctrol_trasaction(&DevInst);
    /* Same transaction API, delivered by broadcast to a 4x4 tile grid. */
    demo_txn_multipl_row(&DevInst);
    return 0;
}
