/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

/**
 * @file aie_runtime_debug.h
 * @brief Runtime debug utilities for AIE tile status inspection.
 *
 * Provides functions to:
 *  - Query and print AIE core status (PC, enabled, stalled, done)
 *  - Read and print DMA BD descriptors (bd_id, addr, len, lock, next, pkt)
 *  - Query DMA channel status for core tiles and shim tiles (S2MM/MM2S)
 *  - Decode current BD from a live channel status register
 *  - Run a verification pass across a set of tile IOs after startup
 */

#ifndef AIE_RUNTIME_DEBUG_H
#define AIE_RUNTIME_DEBUG_H

#include "aie_runtime.h"

/* --------------------------------------------------------------------------
 * AIE generation selector
 *
 * Controls which hardware event name table is used by AieRt_EventName().
 * CORE and MEM module event IDs are identical between AIEML and AIE2PS.
 * PL module event IDs differ (AIE2PS extends to ID 183 vs AIEML max 127).
 *
 * Set g_aiert_gen (or call AieRt_SetAieGen) before using debug APIs if the
 * target device is not AIE2PS.
 * -------------------------------------------------------------------------- */

typedef enum {
    AIERT_GEN_AIEML = 0,  /* AIE-ML / AIE2 */
    AIERT_GEN_AIE2PS = 1, /* AIE2PS — default */
} AieRt_AieGen;

extern AieRt_AieGen g_aiert_gen; /* default: AIERT_GEN_AIE2PS */

/**
 * Set the AIE generation for event name lookups.
 * @param gen  AIERT_GEN_AIEML or AIERT_GEN_AIE2PS.
 */
void AieRt_SetAieGen(AieRt_AieGen gen);

/* --------------------------------------------------------------------------
 * Event name lookup
 *
 * Returns the human-readable name for a hardware-local event ID.
 * IDs are module-local (0-based per module), sourced from:
 *   xaie_events_aieml.h  (XAIEML_EVENTS_{CORE,MEM,PL}_*)
 *   xaie_events_aie2ps.h (XAIE2PS_EVENTS_{CORE,MEM,PL}_*)
 *
 * CORE and MEM tables are identical between AIEML and AIE2PS.
 * PL table differs: AIEML max=127, AIE2PS max=183.
 *
 * @param module  XAIE_CORE_MOD, XAIE_MEM_MOD, or XAIE_PL_MOD.
 * @param id      Module-local hardware event ID.
 * @return        Pointer to a static string with the event name.
 *                Returns "EVT_<id>" for unknown IDs (written to a small
 *                static buffer — not re-entrant for unknown IDs).
 * -------------------------------------------------------------------------- */
const char *AieRt_EventName(XAie_ModuleType module, uint32_t id);

/* --------------------------------------------------------------------------
 * Print all active events from a module's event status register(s)
 * -------------------------------------------------------------------------- */

/**
 * Read the Memory Module event status registers of a core tile and print
 * the name of every set bit using AieRt_EventName(XAIE_MEM_MOD, id).
 * Covers event IDs 0-31 (register 0x00014200) and 32-63 (0x00014204).
 * @param dev   Device instance.
 * @param tile  Core tile location (row >= XAIE_AIE_TILE_ROW_START).
 */
void AieRt_PrintMemModuleEvents(XAie_DevInst *dev, XAie_LocType tile);

/**
 * Call AieRt_PrintMemModuleEvents for every core tile in the list.
 */
void AieRt_PrintMemModuleEventsAll(XAie_DevInst *dev, const XAie_LocType *tiles, uint32_t num_tiles);

/**
 * Read the Core Module event status registers of a core tile and print
 * the name of every set bit using AieRt_EventName(XAIE_CORE_MOD, id).
 * Covers event IDs 0-31 (register 0x00034200) and 32-63 (0x00034204).
 * @param dev   Device instance.
 * @param tile  Core tile location (row >= XAIE_AIE_TILE_ROW_START).
 */
void AieRt_PrintCoreModuleEvents(XAie_DevInst *dev, XAie_LocType tile);

/**
 * Call AieRt_PrintCoreModuleEvents for every core tile in the list.
 */
void AieRt_PrintCoreModuleEventsAll(XAie_DevInst *dev, const XAie_LocType *tiles, uint32_t num_tiles);

/* --------------------------------------------------------------------------
 * Decoded DMA channel status (AIEML / AIE2P register layout)
 *
 *  Status raw word bits (shim NOC module, same layout for core/mem tiles):
 *   [1:0]  STATUS       (0=Idle/Complete, 1=Running, 2=Paused, 3=Paused)
 *   [2]    STALLED_LOCK_ACQ
 *   [3]    STALLED_LOCK_REL
 *   [4]    STALLED_STREAM  (starvation for S2MM, backpressure for MM2S)
 *   [5]    STALLED_TCT_OR_COUNT_FIFO_FULL
 *   [10]   ERROR_BD_UNAVAILABLE   (S2MM only)
 *   [11]   ERROR_BD_INVALID
 *   [12]   ERROR_FOT_LENGTH_EXCEEDED (S2MM only)
 *   [13]   ERROR_FOT_BDS_PER_TASK    (S2MM only)
 *   [16]   AXI_MM_DECODE_ERROR
 *   [17]   AXI_MM_SLAVE_ERROR
 *   [18]   TASK_QUEUE_OVERFLOW
 *   [19]   CHANNEL_RUNNING
 *   [22:20] TASK_QUEUE_SIZE  (number of BDs in queue)
 *   [27:24] CUR_BD           (current BD being processed)
 * -------------------------------------------------------------------------- */

/* Raw bit masks applied to the u32 returned by XAie_DmaGetChannelStatus */
#define AIERT_DMA_STATUS_MASK 0x00000003u
#define AIERT_DMA_STALL_LOCK_ACQ_MASK 0x00000004u
#define AIERT_DMA_STALL_LOCK_REL_MASK 0x00000008u
#define AIERT_DMA_STALL_STREAM_MASK 0x00000010u
#define AIERT_DMA_STALL_TCT_MASK 0x00000020u
#define AIERT_DMA_ERR_BD_UNAVAIL_MASK 0x00000400u
#define AIERT_DMA_ERR_BD_INVALID_MASK 0x00000800u
#define AIERT_DMA_ERR_FOT_LEN_MASK 0x00001000u
#define AIERT_DMA_ERR_FOT_BDS_MASK 0x00002000u
#define AIERT_DMA_ERR_AXI_DECODE_MASK 0x00010000u
#define AIERT_DMA_ERR_AXI_SLAVE_MASK 0x00020000u
#define AIERT_DMA_TASK_Q_OVERFLOW_MASK 0x00040000u
#define AIERT_DMA_CHANNEL_RUNNING_MASK 0x00080000u
#define AIERT_DMA_TASK_Q_SIZE_MASK 0x00700000u
#define AIERT_DMA_TASK_Q_SIZE_LSB 20u
#define AIERT_DMA_CUR_BD_MASK 0x0F000000u
#define AIERT_DMA_CUR_BD_LSB 24u

/* Per-channel register stride (AIEML / AIE2PS: each channel has its own
 * status register at +0x4 from the previous one) */
#define AIERT_DMA_CHNUM_OFFSET 0x4u

/* DMA channel status register base offsets (tile-relative).
 * S2MM_STATUS_0 is the base; ch1 is at base + AIERT_DMA_CHNUM_OFFSET.
 * MM2S_STATUS_0 is offset from S2MM_STATUS_0 by the direction offset.
 *
 * Core tile (MEMORY_MODULE) — same for AIEML and AIE2PS:
 *   S2MM_STATUS_0 = 0x0001DF00   S2MM_STATUS_1 = 0x0001DF04
 *   MM2S_STATUS_0 = 0x0001DF10   MM2S_STATUS_1 = 0x0001DF14
 * Shim NOC tile (NOC_MODULE) — AIEML:
 *   S2MM_STATUS_0 = 0x0001D220   MM2S_STATUS_0 = 0x0001D228
 * Shim NOC tile (NOC_MODULE) — AIE2PS:
 *   S2MM_STATUS_0 = 0x00009320   MM2S_STATUS_0 = 0x00009328
 * Mem tile (MEM_TILE_MODULE) — same for AIEML and AIE2PS:
 *   S2MM_STATUS_0 = 0x000A0660   MM2S_STATUS_0 = 0x000A0680
 */
#define AIERT_DMA_CORE_S2MM_STATUS_BASE 0x0001DF00u
#define AIERT_DMA_CORE_MM2S_STATUS_BASE 0x0001DF10u
#define AIERT_DMA_SHIM_S2MM_STATUS_BASE_ML 0x0001D220u  /* AIEML */
#define AIERT_DMA_SHIM_MM2S_STATUS_BASE_ML 0x0001D228u  /* AIEML */
#define AIERT_DMA_SHIM_S2MM_STATUS_BASE_2PS 0x00009320u /* AIE2PS */
#define AIERT_DMA_SHIM_MM2S_STATUS_BASE_2PS 0x00009328u /* AIE2PS */
#define AIERT_DMA_MEM_S2MM_STATUS_BASE 0x000A0660u
#define AIERT_DMA_MEM_MM2S_STATUS_BASE 0x000A0680u

/* STATUS field values */
#define AIERT_DMA_STATUS_IDLE 0u
#define AIERT_DMA_STATUS_RUNNING 1u
#define AIERT_DMA_STATUS_PAUSED 2u

typedef struct {
    uint8_t status;          /* 0=Idle, 1=Running, 2=Paused */
    uint8_t channel_running; /* CHANNEL_RUNNING bit */
    uint8_t task_q_size;     /* pending BDs in queue */
    uint8_t task_q_overflow;
    uint8_t cur_bd; /* current BD being processed */
    uint8_t stall_lock_acq;
    uint8_t stall_lock_rel;
    uint8_t stall_stream; /* starvation (S2MM) or backpressure (MM2S) */
    uint8_t stall_tct;
    uint8_t err_bd_unavail;
    uint8_t err_bd_invalid;
    uint8_t err_fot_len;
    uint8_t err_fot_bds;
    uint8_t err_axi_decode;
    uint8_t err_axi_slave;
    uint32_t raw; /* raw register value */
} AieRt_DmaChStatusDecoded;

/* --------------------------------------------------------------------------
 * Core tile Memory Module event status register constants
 *
 * Register: core tile Memory Module event status register 0 covers local
 * event IDs 0-31.  Address (from debug.md):
 *   full addr = 0x20000000000 + (col << 25) + (row << 20) + 0x00014200
 *   XAie_Read32(dev, (col<<25)|(row<<20)|0x00014200, &out)
 *
 * Bit positions are the hardware-local event IDs (NOT global aie2psevent.md
 * IDs).  Confirmed example from debug.md:
 *   DMA_S2MM_0_START_TASK = local event 19 = bit 19 of event register 0.
 * Consecutive channel ordering:
 *   bit 19 : DMA_S2MM_0_START_TASK
 *   bit 20 : DMA_S2MM_1_START_TASK
 *   bit 21 : DMA_MM2S_0_START_TASK
 *   bit 22 : DMA_MM2S_1_START_TASK
 *   bit 23 : DMA_S2MM_0_FINISHED_TASK
 *   bit 24 : DMA_S2MM_1_FINISHED_TASK
 *   bit 25 : DMA_MM2S_0_FINISHED_TASK
 *   bit 26 : DMA_MM2S_1_FINISHED_TASK
 * -------------------------------------------------------------------------- */

/* Core module control/status/PC/SP register offsets (tile-relative).
 * XAIEMLGBL_CORE_MODULE_CORE_STATUS = 0x00032004 (mask 0x003FFFFF)
 * XAIEMLGBL_CORE_MODULE_CORE_PC     = 0x00031100 (mask 0x000FFFFF, 20-bit)
 * XAIEMLGBL_CORE_MODULE_CORE_SP     = 0x00031120 */
#define AIERT_CORE_STATUS_REG 0x00038004u
#define AIERT_CORE_PC_REG 0x00032D00
#define AIERT_CORE_SP_REG 0x00032D20
#define AIERT_CORE_PC_MASK 0x000FFFFFu

/* Memory module event status registers (tile-relative, on core tiles).
 *   reg0 (0x00014200): MEM event IDs   0-31
 *   reg1 (0x00014204): MEM event IDs  32-63
 *   reg2 (0x00014208): MEM event IDs  64-95
 *   reg3 (0x0001420C): MEM event IDs  96-127
 */
#define AIERT_CORE_MEM_EVT_STATUS_REG 0x00014200u
#define AIERT_CORE_MEM_EVT_STATUS_REG1 0x00014204u
#define AIERT_CORE_MEM_EVT_STATUS_REG2 0x00014208u
#define AIERT_CORE_MEM_EVT_STATUS_REG3 0x0001420Cu

/* Core module event status registers (tile-relative, on core tiles).
 *   reg0 (0x00034200): Core event IDs   0-31
 *   reg1 (0x00034204): Core event IDs  32-63
 *   reg2 (0x00034208): Core event IDs  64-95
 *   reg3 (0x0003420C): Core event IDs  96-127
 */
#define AIERT_CORE_CORE_EVT_STATUS_REG 0x00034200u
#define AIERT_CORE_CORE_EVT_STATUS_REG1 0x00034204u
#define AIERT_CORE_CORE_EVT_STATUS_REG2 0x00034208u
#define AIERT_CORE_CORE_EVT_STATUS_REG3 0x0003420Cu

/* Shim tile PL module event status register offsets (tile-relative).
 * Same offsets as CORE module event status registers but accessed on
 * shim tiles (row=0). Confirmed from XAIE2PSGBL_PL_MODULE_EVENT_STATUS* /
 * XAIEMLGBL_PL_MODULE_EVENT_STATUS* in xaie2psgbl_params.h / xaiemlgbl_params.h.
 *   reg0 (0x00034200): PL event IDs   0-31
 *   reg1 (0x00034204): PL event IDs  32-63
 *   reg2 (0x00034208): PL event IDs  64-95
 *   reg3 (0x0003420C): PL event IDs  96-127
 *   reg4 (0x00034210): PL event IDs 128-159
 *   reg5 (0x00034214): PL event IDs 160-181 (AIE2PS)
 */
#define AIERT_SHIM_PL_EVT_STATUS_REG0 0x00034200u
#define AIERT_SHIM_PL_EVT_STATUS_REG1 0x00034204u
#define AIERT_SHIM_PL_EVT_STATUS_REG2 0x00034208u
#define AIERT_SHIM_PL_EVT_STATUS_REG3 0x0003420Cu
#define AIERT_SHIM_PL_EVT_STATUS_REG4 0x00034210u
#define AIERT_SHIM_PL_EVT_STATUS_REG5 0x00034214u

/* --------------------------------------------------------------------------
 * Core module event status register bit positions (aie2psevent.md IDs 0-63)
 *
 * Register 0 (AIERT_CORE_CORE_EVT_STATUS_REG,  0x00034200): event IDs  0-31
 * Register 1 (AIERT_CORE_CORE_EVT_STATUS_REG1, 0x00034204): event IDs 32-63
 *   bit N of register 1 = event ID (32 + N)
 *
 * Stall / activity events (AIE2PS Table 4-8, all in register 0):
 *   ID 23 (bit 23) : MEMORY_STALL
 *   ID 24 (bit 24) : STREAM_STALL
 *   ID 25 (bit 25) : CASCADE_STALL
 *   ID 26 (bit 26) : LOCK_STALL
 *   ID 27 (bit 27) : DEBUG_HALTED
 *   ID 28 (bit 28) : ACTIVE
 *   ID 29 (bit 29) : DISABLED
 *   ID 30 (bit 30) : ECC_ERROR_STALL
 *   ID 31 (bit 31) : ECC_SCRUBBING_STALL
 * -------------------------------------------------------------------------- */

/* Bits within register 0 (event ID = bit_pos, AIE2PS layout) */
#define AIERT_CORE_EVT0_MS_STALL_BIT 23u        /* ID 23 — MEMORY_STALL */
#define AIERT_CORE_EVT0_STREAM_STALL_BIT 24u    /* ID 24 */
#define AIERT_CORE_EVT0_CASCADE_STALL_BIT 25u   /* ID 25 */
#define AIERT_CORE_EVT0_LOCK_STALL_BIT 26u      /* ID 26 */
#define AIERT_CORE_EVT0_DEBUG_HALTED_BIT 27u    /* ID 27 */
#define AIERT_CORE_EVT0_ACTIVE_BIT 28u          /* ID 28 */
#define AIERT_CORE_EVT0_DISABLED_BIT 29u        /* ID 29 */
#define AIERT_CORE_EVT0_ECC_ERR_STALL_BIT 30u   /* ID 30 */
#define AIERT_CORE_EVT0_ECC_SCRUB_STALL_BIT 31u /* ID 31 */

#define AIERT_EVT_DMA_S2MM_0_START_BIT 19u
#define AIERT_EVT_DMA_S2MM_1_START_BIT 20u
#define AIERT_EVT_DMA_MM2S_0_START_BIT 21u
#define AIERT_EVT_DMA_MM2S_1_START_BIT 22u
#define AIERT_EVT_DMA_S2MM_0_FINISH_BIT 23u
#define AIERT_EVT_DMA_S2MM_1_FINISH_BIT 24u
#define AIERT_EVT_DMA_MM2S_0_FINISH_BIT 25u
#define AIERT_EVT_DMA_MM2S_1_FINISH_BIT 26u

/* --------------------------------------------------------------------------
 * Core status functions
 * -------------------------------------------------------------------------- */

/**
 * Print AIE core status for a single tile.
 * Reports: enabled, reset, done, error-halt, debug-halt, mem/lock/stream stalls,
 *          PC value (if core is not in reset).
 * @param dev   Device instance.
 * @param tile  Tile location (must be a core tile, row >= XAIE_AIE_TILE_ROW_START).
 */
void AieRt_PrintCoreStatus(XAie_DevInst *dev, XAie_LocType tile);

/**
 * Print core status for every tile in the list that is a core tile.
 * @param dev        Device instance.
 * @param tiles      Array of tile locations.
 * @param num_tiles  Length of the array.
 */
void AieRt_PrintCoreStatusAll(XAie_DevInst *dev, const XAie_LocType *tiles, uint32_t num_tiles);

/* --------------------------------------------------------------------------
 * Core module activity / stall event analysis (Core Module event status reg)
 * -------------------------------------------------------------------------- */

/**
 * Read the core tile Core Module event status registers (0x00034200 and
 * 0x00034204) and print the activity / stall event bits derived from
 * aie2psevent.md IDs 41-55:
 *
 *   ACTIVE, DISABLED, MS_STALL, PM_STALL (instr-load stall),
 *   STREAM_STALL, CASCADE_STALL, LOCK_STALL, ECC_ERROR_STALL,
 *   ECC_SCRUBBING_STALL, STALL_NOP.
 *
 * Address formula (debug.md):
 *   full addr = 0x20000000000 + (col<<25) + (row<<20) + reg_offset
 *   XAie_Read32(dev, (col<<25)|(row<<20)|0x00034200, &out)
 *
 * Warns when the core is DISABLED or when any stall bit is set.
 * @param dev   Device instance.
 * @param tile  Core tile location (row >= XAIE_AIE_TILE_ROW_START).
 */
void AieRt_PrintCoreActivityEvents(XAie_DevInst *dev, XAie_LocType tile);

/**
 * Call AieRt_PrintCoreActivityEvents for every core tile in the list.
 * @param dev        Device instance.
 * @param tiles      Array of tile locations.
 * @param num_tiles  Length of the array.
 */
void AieRt_PrintCoreActivityEventsAll(XAie_DevInst *dev, const XAie_LocType *tiles, uint32_t num_tiles);

/* --------------------------------------------------------------------------
 * DMA BD event analysis (Memory Module event status register)
 * -------------------------------------------------------------------------- */

/**
 * Read the core tile Memory Module event status register (+0x4208) and print
 * the DMA BD start/finish event bits for all four channels (S2MM ch0/ch1,
 * MM2S ch0/ch1).  Warns when a channel started a task but never finished
 * (indicating a stall or incomplete transfer at snapshot time).
 * @param dev   Device instance.
 * @param tile  Core tile location (row >= XAIE_AIE_TILE_ROW_START).
 */
void AieRt_PrintCoreTileDmaBdEvents(XAie_DevInst *dev, XAie_LocType tile);

/**
 * Call AieRt_PrintCoreTileDmaBdEvents for every core tile in the list.
 * @param dev        Device instance.
 * @param tiles      Array of tile locations.
 * @param num_tiles  Length of the array.
 */
void AieRt_PrintCoreTileDmaBdEventsAll(XAie_DevInst *dev, const XAie_LocType *tiles, uint32_t num_tiles);

/* --------------------------------------------------------------------------
 * DMA BD functions
 * -------------------------------------------------------------------------- */

/**
 * Read and print a single DMA BD descriptor from hardware.
 * Prints: addr, len (bytes), lock (acq/rel id and values), next BD, packet.
 * @param dev     Device instance.
 * @param tile    Tile location.
 * @param bd_id   BD index to read.
 */
void AieRt_PrintBdInfo(XAie_DevInst *dev, XAie_LocType tile, uint8_t bd_id);

/**
 * Print all configured BDs for a tile (bd_id 0 .. num_bds-1).
 * Scans all BDs and only prints those with non-zero address or length.
 * @param dev   Device instance.
 * @param tile  Tile location.
 */
void AieRt_PrintAllBds(XAie_DevInst *dev, XAie_LocType tile);

/* --------------------------------------------------------------------------
 * Raw BD register dump (shim tile)
 *
 * Reads all 16 BD slots by directly accessing DMA BD registers via
 * XAie_Read32.  For each BD where word0 (Buffer_Length) is non-zero,
 * prints all words decoded.  Offsets are generation-specific.
 *
 * AIEML Shim NOC tile BD layout (xaiemlgbl_params.h):
 *   Base offset: 0x1D000, each BD is 8 words = 0x20 bytes apart
 *   16 BDs: BD0 @ 0x1D000 .. BD15 @ 0x1D1E0
 *
 * AIE2PS Shim NOC tile BD layout (xaie2psgbl_params.h):
 *   Base offset: 0x9000, each BD is 9 words = 0x30 bytes apart
 *   16 BDs: BD0 @ 0x9000 .. BD15 @ 0x92D0
 *
 *   Word 0 (BD_0): Buffer_Length [31:0]
 *   Word 1 (BD_1): Base_Address_Low [31:0]
 *   Word 2 (BD_2): Enable_Packet [30], Out_of_Order_BD_ID [29:24],
 *                   Packet_ID [23:19], Packet_Type [18:16],
 *                   Base_Address_High [15:0]
 *   Word 3 (BD_3): D0_Stepsize [19:0], D0_Wrap [29:20]
 *   Word 4 (BD_4): D1_Stepsize [19:0], D1_Wrap [29:20]
 *   Word 5 (BD_5): D2_Stepsize [19:0], D2_Wrap [29:20]
 *   Word 6 (BD_6): Iteration_Stepsize [19:0], Iteration_Wrap [25:20],
 *                   Iteration_Current [31:26]
 *   Word 7 (BD_7 / AIE2PS BD_7):
 *     AIE2PS: D3_Stepsize [19:0], D3_Wrap [29:20]
 *     AIEML:  Valid_BD [0], Next_BD [4:1], Use_Next_BD [5],
 *             Lock_Acq_Value [13:6], Lock_Acq_ID [17:14],
 *             Lock_Rel_Value [25:18], Lock_Rel_ID [29:26],
 *             Lock_Acq_Enable [30], Lock_Rel_Enable [31]
 *   Word 8 (BD_8, AIE2PS only):
 *     Valid_BD [0], Next_BD [4:1], Use_Next_BD [5],
 *     Lock_Acq_Value [13:6], Lock_Acq_ID [17:14],
 *     Lock_Rel_Value [25:18], Lock_Rel_ID [29:26],
 *     Lock_Acq_Enable [30], Lock_Rel_Enable [31]
 * -------------------------------------------------------------------------- */

/* AIEML shim BD layout */
#define AIERT_SHIM_BD_BASE_ML 0x0001D000u /* BD0 word0 offset (AIEML) */
#define AIERT_SHIM_BD_STRIDE_ML 0x20u     /* 8 words * 4 bytes per BD (AIEML) */
#define AIERT_SHIM_BD_WORDS_ML 8u         /* 8 x 32-bit words per BD (AIEML) */

/* AIE2PS shim BD layout */
#define AIERT_SHIM_BD_BASE_2PS 0x00009000u /* BD0 word0 offset (AIE2PS) */
#define AIERT_SHIM_BD_STRIDE_2PS 0x30u     /* 9 words * 4 bytes per BD (AIE2PS) */
#define AIERT_SHIM_BD_WORDS_2PS 9u         /* 9 x 32-bit words per BD (AIE2PS) */

/* Legacy names — default to AIEML for backward compat with existing callers */
#define AIERT_SHIM_BD_BASE AIERT_SHIM_BD_BASE_ML
#define AIERT_SHIM_BD_STRIDE AIERT_SHIM_BD_STRIDE_ML
#define AIERT_SHIM_BD_COUNT 16u
#define AIERT_SHIM_BD_WORDS AIERT_SHIM_BD_WORDS_ML

/**
 * Dump all 16 raw BD registers for a shim tile.
 * Only prints BDs where word0 (Buffer_Length) is non-zero.
 * Decodes stride/wrap for D0-D2, iteration, lock, next BD, packet fields.
 * @param dev  Device instance.
 * @param col  Column of the shim tile (row=0).
 */
void AieRt_PrintShimBdRawAll(XAie_DevInst *dev, uint8_t col);

/* --------------------------------------------------------------------------
 * DMA channel status functions
 * -------------------------------------------------------------------------- */

/**
 * Read the raw DMA channel status register via XAie_Read32 without masking.
 *
 * Unlike XAie_DmaGetChannelStatus (which masks the raw value, stripping
 * STATUS[1:0], CUR_BD[27:24], and error bits), this function returns the
 * full unmasked register content so AieRt_DecodeDmaChStatus can decode
 * all fields correctly.
 *
 * Targets AIEML (aie version 5) and AIE2PS only.  Register addresses are
 * selected based on tile type (core, shim, memtile).
 *
 * @param dev     Device instance.
 * @param tile    Tile location.
 * @param ch      Channel number (0 or 1).
 * @param dir     DMA_S2MM or DMA_MM2S.
 * @param status  [out] Raw unmasked 32-bit status register value.
 * @return        XAIE_OK on success, or an error code.
 */
AieRC AieRt_DmaGetChannelStatusFull(XAie_DevInst *dev, XAie_LocType tile, uint8_t ch, XAie_DmaDirection dir,
                                    uint32_t *status);

/**
 * Decode a raw channel status word into AieRt_DmaChStatusDecoded.
 * @param raw_status  Value returned by AieRt_DmaGetChannelStatusFull.
 * @return Decoded status struct.
 */
AieRt_DmaChStatusDecoded AieRt_DecodeDmaChStatus(uint32_t raw_status);

/**
 * Print decoded DMA channel status for one channel on a tile.
 * If the channel is running, also reads and prints the current BD contents.
 * @param dev     Device instance.
 * @param tile    Tile location.
 * @param ch      Channel number.
 * @param dir     DMA_S2MM or DMA_MM2S.
 */
void AieRt_PrintDmaChStatus(XAie_DevInst *dev, XAie_LocType tile, uint8_t ch, XAie_DmaDirection dir);

/**
 * Print S2MM and MM2S status for channels 0 and 1 of a shim tile (row=0).
 * @param dev  Device instance.
 * @param col  Column of the shim tile.
 */
void AieRt_PrintShimDmaStatus(XAie_DevInst *dev, uint8_t col);

/**
 * Read the shim tile PL Module event status registers (0x00034200-0x0003420C)
 * and print the name of every set bit using AieRt_EventName(XAIE_PL_MOD, id).
 * Covers event IDs 0-127 (registers 0-3). Flags DMA and error events.
 * @param dev  Device instance.
 * @param col  Column of the shim tile (row=0).
 */
void AieRt_PrintShimPlModuleEvents(XAie_DevInst *dev, uint8_t col);

/**
 * Print DMA channel status for all channels of a core tile.
 * Reports S2MM ch0+ch1 and MM2S ch0+ch1.
 * @param dev   Device instance.
 * @param tile  Core tile location.
 */
void AieRt_PrintCoreTileDmaStatus(XAie_DevInst *dev, XAie_LocType tile);

/* --------------------------------------------------------------------------
 * Verification
 * -------------------------------------------------------------------------- */

/**
 * Verify a set of struct_io descriptors against live hardware state.
 *
 * For each io:
 *  - Reads the DMA channel status register.
 *  - Checks that the channel reported as running/done (not stalled, no errors).
 *  - Reads the BD that was registered for this IO and prints its contents.
 *
 * Prints PASS or FAIL for each IO and returns the total fail count.
 *
 * @param dev       Device instance.
 * @param ios       Array of struct_io descriptors (from __Runtime_dma_createio_4).
 * @param num_ios   Length of the array.
 * @return          Number of failed checks (0 = all pass).
 */
int AieRt_VerifyIoDescriptors(XAie_DevInst *dev, const struct_io *ios, uint32_t num_ios);

/**
 * Convenience: print full debug snapshot for a GEMM-style run.
 * Prints core status for all core tiles, shim DMA status for all shim columns
 * referenced by the ios, and DMA status for all core tiles referenced by ios.
 * @param dev       Device instance.
 * @param ios       Array of struct_io descriptors.
 * @param num_ios   Length of the ios array.
 * @param tiles     Core tile array (from kernel_group).
 * @param num_tiles Length of the tiles array.
 */
void AieRt_DebugSnapshot(XAie_DevInst *dev, const struct_io *ios, uint32_t num_ios, const XAie_LocType *tiles,
                         uint32_t num_tiles);

/**
 * Convenience wrapper: build struct_io array from raw coordinate arrays and call AieRt_DebugSnapshot.
 * @param dev        Device instance.
 * @param io_cols    Column for each IO (length num_ios).
 * @param io_rows    Row for each IO.
 * @param io_chs     Channel ID for each IO.
 * @param io_bds     BD ID for each IO.
 * @param io_dirs    DMA direction for each IO (XAie_DmaDirection, cast from int).
 * @param num_ios    Number of IOs.
 * @param tile_cols  Column for each core tile (length num_tiles); may be NULL if num_tiles==0.
 * @param tile_rows  Row for each core tile; may be NULL if num_tiles==0.
 * @param num_tiles  Number of core tiles.
 */
void AieRt_DebugSnapshotFromCoords(XAie_DevInst *dev, const uint8_t *io_cols, const uint8_t *io_rows,
                                   const uint8_t *io_chs, const uint8_t *io_bds, const int *io_dirs, uint32_t num_ios,
                                   const uint8_t *tile_cols, const uint8_t *tile_rows, uint32_t num_tiles);

/* --------------------------------------------------------------------------
 * Stream switch register debug
 *
 * Register layout (tile-relative offsets, AIEML, xaiemlgbl_params.h):
 *
 * Core tile (row >= XAIE_AIE_TILE_ROW_START) — CORE_MODULE:
 *   Master config base  : 0x0003F000  (+4 per port, 23 ports)
 *   Slave config base   : 0x0003F100  (+4 per port, 25 slave ports)
 *   Slave slot base     : 0x0003F200  (+16 per slave-port, 4 slots each at +0/+4/+8/+C)
 *   Source: XAIE2PSGBL_CORE_MODULE_STREAM_SWITCH_MASTER_CONFIG_AIE_CORE0
 *
 * MemTile (row 1-2, XAIE_RES_TILE_ROW_START .. +XAIE_RES_TILE_NUM_ROWS-1) — MEM_TILE_MODULE:
 *   Master config base  : 0x000B0000  (+4 per port, 19 ports: DMA0-5, CTRL, NORTH0-5, SOUTH0-5)
 *   Slave config base   : 0x000B0100  (+4 per port, 20 ports: DMA0-5, CTRL, NORTH0-5, SOUTH0-5, TRACE)
 *   Slave slot base     : 0x000B0200  (+16 per slave-port, 4 slots each)
 *   Source: XAIE2PSGBL_MEM_TILE_MODULE_STREAM_SWITCH_MASTER_CONFIG_DMA0
 *
 * Shim tile (row == 0) — PL_MODULE:
 *   Master config base  : 0x0003F000  (same region as core tile)
 *   Slave config base   : 0x0003F100
 *   Slave slot base     : 0x0003F200
 *   Source: XAIE2PSGBL_PL_MODULE_STREAM_SWITCH_MASTER_CONFIG_TILE_CTRL
 *
 * Master config register (32-bit):
 *   [31] MASTER_ENABLE   [30] PACKET_ENABLE   [7] DROP_HEADER   [6:0] CONFIG/ARBITOR
 *
 * Slave config register (32-bit):
 *   [31] SLAVE_ENABLE
 *
 * Slave slot register (32-bit, from xaiemlgbl_params.h SLAVE_AIE_CORE0_SLOT0_*):
 *   [28:24] ID/PktId   [20:16] MASK   [8] ENABLE   [5:4] MSEL   [2:0] ARBIT
 * -------------------------------------------------------------------------- */

/* Tile-relative stream switch register base offsets.
 *
 * Core tile (CORE_MODULE) and Shim tile (PL_MODULE) share the same offsets:
 *   Source: XAIE2PSGBL_CORE_MODULE_STREAM_SWITCH_MASTER_CONFIG_AIE_CORE0 = 0x0003F000
 *           XAIE2PSGBL_PL_MODULE_STREAM_SWITCH_MASTER_CONFIG_TILE_CTRL   = 0x0003F000
 *
 * MemTile (MEM_TILE_MODULE) uses a completely different register region:
 *   Source: XAIE2PSGBL_MEM_TILE_MODULE_STREAM_SWITCH_MASTER_CONFIG_DMA0  = 0x000B0000
 *
 * All confirmed from xaie2psgbl_params.h / xaie2psgbl_reginit.c.
 */
#define AIERT_SS_MASTER_BASE 0x0003F000u /* core and shim */
#define AIERT_SS_SLAVE_BASE 0x0003F100u  /* core and shim */
#define AIERT_SS_SLOT_BASE 0x0003F200u   /* core and shim */

#define AIERT_SS_MEM_MASTER_BASE 0x000B0000u /* mem tile only */
#define AIERT_SS_MEM_SLAVE_BASE 0x000B0100u  /* mem tile only */
#define AIERT_SS_MEM_SLOT_BASE 0x000B0200u   /* mem tile only */

/* Master config bit masks */
#define AIERT_SS_MSTR_ENABLE_MASK 0x80000000u
#define AIERT_SS_MSTR_PKT_EN_MASK 0x40000000u
#define AIERT_SS_MSTR_DROP_MASK 0x00000080u
#define AIERT_SS_MSTR_CONFIG_MASK 0x0000007Fu

/* Slave config bit masks */
#define AIERT_SS_SLV_ENABLE_MASK 0x80000000u

/* Slave slot bit masks (from xaiemlgbl_params.h SLAVE_AIE_CORE0_SLOT0_* fields):
 *   [28:24] ID    (PktId)  mask 0x1F000000
 *   [20:16] MASK  (PktMask) mask 0x001F0000
 *   [8]     ENABLE (slot valid) mask 0x00000100
 *   [5:4]   MSEL   mask 0x00000030
 *   [2:0]   ARBIT  mask 0x00000007
 */
#define AIERT_SS_SLOT_ENABLE_MASK 0x00000100u /* slot valid / enabled */
#define AIERT_SS_SLOT_PKTID_MASK 0x1F000000u
#define AIERT_SS_SLOT_PKTID_LSB 24u
#define AIERT_SS_SLOT_MASK_MASK 0x001F0000u
#define AIERT_SS_SLOT_MASK_LSB 16u
#define AIERT_SS_SLOT_MSEL_MASK 0x00000030u
#define AIERT_SS_SLOT_MSEL_LSB 4u
#define AIERT_SS_SLOT_ARB_MASK 0x00000007u
#define AIERT_SS_SLOT_ARB_LSB 0u

/**
 * Print all stream switch master config, slave config, and slave slot registers
 * for a single tile by directly reading hardware via XAie_Read32.
 *
 * For each master port: prints enable, packet-switch mode, drop-header, and
 * the arbitor/configuration field.
 * For each slave port: prints enable, and for each slot prints PktId, Mask,
 * MSel, Arbitor.
 *
 * Only non-zero / enabled registers are printed to keep output concise.
 * Set print_all=1 to print every register regardless of value.
 *
 * Tile type is detected automatically:
 *   row == 0                              → SHIM  (22 master, 23 slave)
 *   row in [XAIE_RES_TILE_ROW_START, +2) → MEMTILE (19 master, 20 slave)
 *   row >= XAIE_AIE_TILE_ROW_START       → CORE  (23 master, 25 slave)
 *
 * @param dev        Device instance.
 * @param tile       Tile location (shim, memtile, or core).
 * @param print_all  0 = print only enabled/non-zero; 1 = print all registers.
 */
void AieRt_PrintStreamSwitchConfig(XAie_DevInst *dev, XAie_LocType tile, int print_all);

/**
 * Call AieRt_PrintStreamSwitchConfig for every tile in the list.
 * @param dev        Device instance.
 * @param tiles      Array of tile locations.
 * @param num_tiles  Length of the array.
 * @param print_all  Passed through to AieRt_PrintStreamSwitchConfig.
 */
void AieRt_PrintStreamSwitchConfigAll(XAie_DevInst *dev, const XAie_LocType *tiles, uint32_t num_tiles, int print_all);

/* --------------------------------------------------------------------------
 * Performance counter debug
 * -------------------------------------------------------------------------- */

/**
 * Read and print memory module perf counter 0 and 1 values for a single
 * core tile.  These are typically armed by the AIE_DEBUG_FLAG_MM2SBDFINISH_COUNTER
 * partition init (__Runtime_perfcnt_setup_mm2s_bd_finished_partition:
 * counter 0 = MM2S ch0 FINISHED_BD, counter 1 = MM2S ch1 FINISHED_BD).
 *
 * @param dev   Device instance.
 * @param tile  Core tile location (row >= XAIE_AIE_TILE_ROW_START).
 */
void AieRt_PrintCoreTilePerfCounters(XAie_DevInst *dev, XAie_LocType tile);

/**
 * Call AieRt_PrintCoreTilePerfCounters for every core tile in the list.
 */
void AieRt_PrintCoreTilePerfCountersAll(XAie_DevInst *dev, const XAie_LocType *tiles, uint32_t num_tiles);

/* --------------------------------------------------------------------------
 * Shim DMA Loopback
 *
 * Self-contained DDR→ShimDMA→DDR loopback test. Allocates src/dst DDR
 * buffers, configures a circuit-switch loopback on SOUTH port 3 of the
 * shim tile, runs a DMA transfer through it, and verifies the data.
 * Useful for validating that the shim DMA + NoC + DDR path is functional
 * on a given column before running the full GEMM pipeline.
 *
 * Reference: aieml-tests/src/xaie_shimdma_loopback.c
 * -------------------------------------------------------------------------- */

/**
 * Run a shim DMA loopback test on a single column.
 *
 * Caller provides pre-allocated DDR src and dst buffers. The function
 * configures a circuit-switch loopback on SOUTH port 3, DMA-reads from
 * srcaddr into the stream, loops it back, and DMA-writes to dstaddr.
 * After transfer, compares srcaddr[] vs dstaddr[] word-by-word.
 *
 * @param dev      Device instance (must be initialized and partitioned).
 * @param col      Shim tile column (must have NOC-DDR path enabled).
 * @param srcaddr  Source DDR buffer (filled by caller with test data).
 * @param dstaddr  Destination DDR buffer (cleared by caller, receives loopback data).
 * @param len      Transfer size in bytes (must be multiple of 4).
 * @return         0 on success (PASS), -2 timeout, -3 mismatch.
 */
int AieRt_ShimDmaLoopback(XAie_DevInst *dev, uint8_t col, uint32_t *srcaddr, uint32_t *dstaddr, uint32_t len);

/**
 * Run shim DMA loopback on multiple columns.
 *
 * @param dev      Device instance.
 * @param cols     Array of column indices to test.
 * @param num_cols Length of the cols array.
 * @param srcaddr  Source DDR buffer (filled by caller).
 * @param dstaddr  Destination DDR buffer (cleared by caller).
 * @param len      Transfer size in bytes (must be multiple of 4).
 * @return         Number of failed columns (0 = all passed).
 */
int AieRt_ShimDmaLoopbackAllCols(XAie_DevInst *dev, const uint8_t *cols, int num_cols, uint32_t *srcaddr,
                                 uint32_t *dstaddr, uint32_t len);

#endif /* AIE_RUNTIME_DEBUG_H */
