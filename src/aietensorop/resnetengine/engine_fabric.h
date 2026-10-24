/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * engine_fabric.h -- the fixed AIE fabric the persistent conv engine runs on.
 *
 * Shared by the kernel (AIE core) and the host driver, so it is plain C with no
 * includes. Everything here is a property of fabric/routing.cc, which is the
 * routing aiehlc generated for src/aietensorop/conv2dstem/conv2dstem.cc
 * (partition {2,6}: absolute cols 2..6, control spine on col 2, 4x4 data mesh
 * on cols 3..6 / rows 3..6). Keep the tables below in sync with that file.
 *
 *   input   shim (k,0) MM2S ch1 -> broadcast to mesh row k-1   (core S2MM ch0)
 *   weights shim (k,0) MM2S ch0 -> broadcast to mesh col k-1   (core S2MM ch1)
 *   output  core MM2S ch0, packet-switched, merged per mesh row into one shim
 *           S2MM channel in out-of-order mode (the packet header names the
 *           shim BD): rows 3,4 -> shim (4,0) ch0,ch1; rows 5,6 -> (3,0) ch0,ch1
 *
 * Columns are PARTITION-RELATIVE (rel col 1..4 = absolute 3..6), as the runtime
 * and routing.cc address them.
 ******************************************************************************/
#ifndef ENGINE_FABRIC_H
#define ENGINE_FABRIC_H

#define EF_PART_START_COL 2
#define EF_PART_NUM_COLS 5
#define EF_MESH_ROWS 4
#define EF_MESH_COLS 4
#define EF_FIRST_ROW 3 /* physical row of mesh row 0 */
#define EF_FIRST_COL 1 /* partition-relative col of mesh col 0 */

/* Core-tile DMA plan (hardware BD ids and lock ids; kernel lock intrinsics use
 * id + EF_LOCK_BASE). Same assignment aiehlc's KERNELCONFIGOFFLOAD used, so the
 * fabric's expectations (packet ids on MM2S ch0) are unchanged. */
#define EF_LOCK_BASE 48
#define EF_BD_IN_PING 0 /* S2MM ch0: input  */
#define EF_BD_IN_PONG 1
#define EF_LK_IN_EMPTY 0
#define EF_LK_IN_FULL 1
#define EF_BD_W_PING 2 /* S2MM ch1: weights */
#define EF_BD_W_PONG 3
#define EF_LK_W_EMPTY 2
#define EF_LK_W_FULL 3
#define EF_BD_OUT_PING 4 /* MM2S ch0: output (packets) */
#define EF_BD_OUT_PONG 5
#define EF_LK_OUT_EMPTY 4
#define EF_LK_OUT_FULL 5

/* Output packet id and out-of-order shim BD per core, [mesh row][mesh col].
 * Read off the KCO section aiehlc generated for this fabric. */
#define EF_PKT_ID_TABLE                                                                                                \
    {                                                                                                                  \
        {4, 5, 6, 7}, {8, 9, 10, 11}, {12, 13, 14, 15}, { 17, 18, 19, 20 }                                             \
    }
#define EF_OOO_BD_TABLE                                                                                                \
    {                                                                                                                  \
        {2, 3, 4, 5}, {7, 8, 9, 10}, {3, 4, 5, 6}, { 8, 9, 10, 11 }                                                    \
    }
/* Shim drain channel per mesh row: rel col and S2MM channel. */
#define EF_OUT_SHIM_COL_TABLE {4, 4, 3, 3}
#define EF_OUT_SHIM_CH_TABLE {0, 1, 0, 1}

#endif
