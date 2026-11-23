/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/
#include <xaiengine.h>
XAie_DevInst *getOrCreateDeviceInstance();

void routing() {
    bool v1 = true;

    // round is 0 hw split in : row -----------
    if (v1) {
        int32_t v2 = XAie_EnableShimDmaToAieStrmPort(getOrCreateDeviceInstance(), XAie_TileLoc(2, 0), 3);
        int32_t v3 = XAie_StrmConnCctEnable(getOrCreateDeviceInstance(), XAie_TileLoc(2, 0), SOUTH, 3, NORTH, 0);
        int32_t v4 = XAie_StrmConnCctEnable(getOrCreateDeviceInstance(), XAie_TileLoc(2, 1), SOUTH, 0, NORTH, 0);
        int32_t v5 = XAie_StrmConnCctEnable(getOrCreateDeviceInstance(), XAie_TileLoc(2, 2), SOUTH, 0, NORTH, 0);
        int32_t v6 = XAie_StrmConnCctEnable(getOrCreateDeviceInstance(), XAie_TileLoc(2, 3), SOUTH, 0, WEST, 0);
        int32_t v7 = XAie_StrmConnCctEnable(getOrCreateDeviceInstance(), XAie_TileLoc(1, 3), EAST, 0, WEST, 0);
        int32_t v8 = XAie_StrmConnCctEnable(getOrCreateDeviceInstance(), XAie_TileLoc(1, 3), EAST, 0, DMA, 0);
        int32_t v9 = XAie_StrmConnCctEnable(getOrCreateDeviceInstance(), XAie_TileLoc(0, 3), EAST, 0, DMA, 0);
    }

    // round is 1 hw split in : row -----------
    if (v1) {
        int32_t v10 = XAie_EnableShimDmaToAieStrmPort(getOrCreateDeviceInstance(), XAie_TileLoc(2, 0), 7);
        int32_t v11 = XAie_StrmConnCctEnable(getOrCreateDeviceInstance(), XAie_TileLoc(2, 0), SOUTH, 7, NORTH, 1);
        int32_t v12 = XAie_StrmConnCctEnable(getOrCreateDeviceInstance(), XAie_TileLoc(2, 1), SOUTH, 1, NORTH, 1);
        int32_t v13 = XAie_StrmConnCctEnable(getOrCreateDeviceInstance(), XAie_TileLoc(2, 2), SOUTH, 1, NORTH, 1);
        int32_t v14 = XAie_StrmConnCctEnable(getOrCreateDeviceInstance(), XAie_TileLoc(2, 3), SOUTH, 1, WEST, 1);
        int32_t v15 = XAie_StrmConnCctEnable(getOrCreateDeviceInstance(), XAie_TileLoc(1, 3), EAST, 1, WEST, 1);
        int32_t v16 = XAie_StrmConnCctEnable(getOrCreateDeviceInstance(), XAie_TileLoc(1, 3), EAST, 1, NORTH, 0);
        int32_t v17 = XAie_StrmConnCctEnable(getOrCreateDeviceInstance(), XAie_TileLoc(0, 3), EAST, 1, NORTH, 0);
        int32_t v18 = XAie_StrmConnCctEnable(getOrCreateDeviceInstance(), XAie_TileLoc(0, 4), SOUTH, 0, DMA, 0);
        int32_t v19 = XAie_StrmConnCctEnable(getOrCreateDeviceInstance(), XAie_TileLoc(1, 4), SOUTH, 0, DMA, 0);
    }

    // round is 0 hw split in : row -----------
    if (v1) {
        int32_t v20 = XAie_EnableShimDmaToAieStrmPort(getOrCreateDeviceInstance(), XAie_TileLoc(3, 0), 3);
        int32_t v21 = XAie_StrmConnCctEnable(getOrCreateDeviceInstance(), XAie_TileLoc(3, 0), SOUTH, 3, NORTH, 0);
        int32_t v22 = XAie_StrmConnCctEnable(getOrCreateDeviceInstance(), XAie_TileLoc(3, 1), SOUTH, 0, NORTH, 0);
        int32_t v23 = XAie_StrmConnCctEnable(getOrCreateDeviceInstance(), XAie_TileLoc(3, 2), SOUTH, 0, NORTH, 0);
        int32_t v24 = XAie_StrmConnCctEnable(getOrCreateDeviceInstance(), XAie_TileLoc(3, 3), SOUTH, 0, WEST, 0);
        int32_t v25 = XAie_StrmConnCctEnable(getOrCreateDeviceInstance(), XAie_TileLoc(2, 3), EAST, 0, WEST, 2);
        int32_t v26 = XAie_StrmConnCctEnable(getOrCreateDeviceInstance(), XAie_TileLoc(1, 3), EAST, 2, WEST, 2);
        int32_t v27 = XAie_StrmConnCctEnable(getOrCreateDeviceInstance(), XAie_TileLoc(1, 3), EAST, 2, DMA, 1);
        int32_t v28 = XAie_StrmConnCctEnable(getOrCreateDeviceInstance(), XAie_TileLoc(0, 3), EAST, 2, DMA, 1);
    }

    // round is 1 hw split in : row -----------
    if (v1) {
        int32_t v29 = XAie_EnableShimDmaToAieStrmPort(getOrCreateDeviceInstance(), XAie_TileLoc(3, 0), 7);
        int32_t v30 = XAie_StrmConnCctEnable(getOrCreateDeviceInstance(), XAie_TileLoc(3, 0), SOUTH, 7, NORTH, 1);
        int32_t v31 = XAie_StrmConnCctEnable(getOrCreateDeviceInstance(), XAie_TileLoc(3, 1), SOUTH, 1, NORTH, 1);
        int32_t v32 = XAie_StrmConnCctEnable(getOrCreateDeviceInstance(), XAie_TileLoc(3, 2), SOUTH, 1, NORTH, 1);
        int32_t v33 = XAie_StrmConnCctEnable(getOrCreateDeviceInstance(), XAie_TileLoc(3, 3), SOUTH, 1, WEST, 1);
        int32_t v34 = XAie_StrmConnCctEnable(getOrCreateDeviceInstance(), XAie_TileLoc(2, 3), EAST, 1, WEST, 3);
        int32_t v35 = XAie_StrmConnCctEnable(getOrCreateDeviceInstance(), XAie_TileLoc(1, 3), EAST, 3, WEST, 3);
        int32_t v36 = XAie_StrmConnCctEnable(getOrCreateDeviceInstance(), XAie_TileLoc(1, 3), EAST, 3, NORTH, 1);
        int32_t v37 = XAie_StrmConnCctEnable(getOrCreateDeviceInstance(), XAie_TileLoc(0, 3), EAST, 3, NORTH, 1);
        int32_t v38 = XAie_StrmConnCctEnable(getOrCreateDeviceInstance(), XAie_TileLoc(0, 4), SOUTH, 1, DMA, 1);
        int32_t v39 = XAie_StrmConnCctEnable(getOrCreateDeviceInstance(), XAie_TileLoc(1, 4), SOUTH, 1, DMA, 1);
    }

    // round is 0 hw split in : row -----------
    if (v1) {
        int32_t v40 = XAie_StrmPktSwSlaveSlotEnable(getOrCreateDeviceInstance(), XAie_TileLoc(0, 3), DMA, 0, 0,
                                                    XAie_PacketInit(9, 0), 31, 0, 0);
        int32_t v41 = XAie_StrmPktSwSlavePortEnable(getOrCreateDeviceInstance(), XAie_TileLoc(0, 3), DMA, 0);
        int32_t v42 = XAie_StrmPktSwMstrPortEnable(getOrCreateDeviceInstance(), XAie_TileLoc(0, 3), EAST, 0,
                                                   XAIE_SS_PKT_DONOT_DROP_HEADER, 0, 1);
        int32_t v43 = XAie_StrmPktSwSlaveSlotEnable(getOrCreateDeviceInstance(), XAie_TileLoc(1, 3), WEST, 0, 0,
                                                    XAie_PacketInit(0, 0), 0, 0, 0);
        int32_t v44 = XAie_StrmPktSwSlavePortEnable(getOrCreateDeviceInstance(), XAie_TileLoc(1, 3), WEST, 0);
        int32_t v45 = XAie_StrmPktSwSlaveSlotEnable(getOrCreateDeviceInstance(), XAie_TileLoc(1, 3), DMA, 0, 0,
                                                    XAie_PacketInit(10, 0), 31, 0, 0);
        int32_t v46 = XAie_StrmPktSwSlavePortEnable(getOrCreateDeviceInstance(), XAie_TileLoc(1, 3), DMA, 0);
        int32_t v47 = XAie_EnableAieToShimDmaStrmPort(getOrCreateDeviceInstance(), XAie_TileLoc(2, 0), 1);
        int32_t v48 = XAie_StrmPktSwMstrPortEnable(getOrCreateDeviceInstance(), XAie_TileLoc(1, 3), EAST, 0,
                                                   XAIE_SS_PKT_DROP_HEADER, 0, 1);
        int32_t v49 = XAie_StrmConnCctEnable(getOrCreateDeviceInstance(), XAie_TileLoc(2, 3), WEST, 0, SOUTH, 0);
        int32_t v50 = XAie_StrmConnCctEnable(getOrCreateDeviceInstance(), XAie_TileLoc(2, 2), NORTH, 0, SOUTH, 0);
        int32_t v51 = XAie_StrmConnCctEnable(getOrCreateDeviceInstance(), XAie_TileLoc(2, 1), NORTH, 0, SOUTH, 0);
        int32_t v52 = XAie_StrmConnCctEnable(getOrCreateDeviceInstance(), XAie_TileLoc(2, 0), NORTH, 0, SOUTH, 1);
    }

    // round is 1 hw split in : row -----------
    if (v1) {
        int32_t v53 = XAie_StrmPktSwSlaveSlotEnable(getOrCreateDeviceInstance(), XAie_TileLoc(0, 4), DMA, 0, 0,
                                                    XAie_PacketInit(11, 0), 31, 0, 0);
        int32_t v54 = XAie_StrmPktSwSlavePortEnable(getOrCreateDeviceInstance(), XAie_TileLoc(0, 4), DMA, 0);
        int32_t v55 = XAie_StrmPktSwMstrPortEnable(getOrCreateDeviceInstance(), XAie_TileLoc(0, 4), EAST, 0,
                                                   XAIE_SS_PKT_DONOT_DROP_HEADER, 0, 1);
        int32_t v56 = XAie_StrmPktSwSlaveSlotEnable(getOrCreateDeviceInstance(), XAie_TileLoc(1, 4), WEST, 0, 0,
                                                    XAie_PacketInit(0, 0), 0, 0, 0);
        int32_t v57 = XAie_StrmPktSwSlavePortEnable(getOrCreateDeviceInstance(), XAie_TileLoc(1, 4), WEST, 0);
        int32_t v58 = XAie_StrmPktSwSlaveSlotEnable(getOrCreateDeviceInstance(), XAie_TileLoc(1, 4), DMA, 0, 0,
                                                    XAie_PacketInit(12, 0), 31, 0, 0);
        int32_t v59 = XAie_StrmPktSwSlavePortEnable(getOrCreateDeviceInstance(), XAie_TileLoc(1, 4), DMA, 0);
        int32_t v60 = XAie_EnableAieToShimDmaStrmPort(getOrCreateDeviceInstance(), XAie_TileLoc(2, 0), 3);
        int32_t v61 = XAie_StrmPktSwMstrPortEnable(getOrCreateDeviceInstance(), XAie_TileLoc(1, 4), SOUTH, 0,
                                                   XAIE_SS_PKT_DROP_HEADER, 0, 1);
        int32_t v62 = XAie_StrmConnCctEnable(getOrCreateDeviceInstance(), XAie_TileLoc(1, 3), NORTH, 0, EAST, 1);
        int32_t v63 = XAie_StrmConnCctEnable(getOrCreateDeviceInstance(), XAie_TileLoc(2, 3), WEST, 1, SOUTH, 1);
        int32_t v64 = XAie_StrmConnCctEnable(getOrCreateDeviceInstance(), XAie_TileLoc(2, 2), NORTH, 1, SOUTH, 1);
        int32_t v65 = XAie_StrmConnCctEnable(getOrCreateDeviceInstance(), XAie_TileLoc(2, 1), NORTH, 1, SOUTH, 1);
        int32_t v66 = XAie_StrmConnCctEnable(getOrCreateDeviceInstance(), XAie_TileLoc(2, 0), NORTH, 1, SOUTH, 3);
    }
    return;
}

#include <xaiengine.h>
XAie_DevInst *getOrCreateDeviceInstance();
