#include <xaiengine.h>

void routing(XAie_DevInst *v1) {
    bool v2 = true;

    { //----routing creation in col ----start-------

        // round is 0 hw split in : col -----------
        if (v2) {
            int32_t v3 = XAie_EnableShimDmaToAieStrmPort(v1, XAie_TileLoc(1, 0), 3);
            int32_t v4 = XAie_StrmConnCctEnable(v1, XAie_TileLoc(1, 0), SOUTH, 3, NORTH, 0);
            int32_t v5 = XAie_StrmConnCctEnable(v1, XAie_TileLoc(1, 1), SOUTH, 0, NORTH, 0);
            int32_t v6 = XAie_StrmConnCctEnable(v1, XAie_TileLoc(1, 2), SOUTH, 0, NORTH, 0);
            int32_t v7 = XAie_StrmConnCctEnable(v1, XAie_TileLoc(1, 3), SOUTH, 0, NORTH, 0);
            int32_t v8 = XAie_StrmConnCctEnable(v1, XAie_TileLoc(1, 3), SOUTH, 0, DMA, 1);
            int32_t v9 = XAie_StrmConnCctEnable(v1, XAie_TileLoc(1, 4), SOUTH, 0, NORTH, 0);
            int32_t v10 = XAie_StrmConnCctEnable(v1, XAie_TileLoc(1, 4), SOUTH, 0, DMA, 1);
            int32_t v11 = XAie_StrmConnCctEnable(v1, XAie_TileLoc(1, 5), SOUTH, 0, NORTH, 0);
            int32_t v12 = XAie_StrmConnCctEnable(v1, XAie_TileLoc(1, 5), SOUTH, 0, DMA, 1);
            int32_t v13 = XAie_StrmConnCctEnable(v1, XAie_TileLoc(1, 6), SOUTH, 0, DMA, 1);
        }

        // round is 1 hw split in : col -----------
        if (v2) {
            int32_t v14 = XAie_EnableShimDmaToAieStrmPort(v1, XAie_TileLoc(1, 0), 7);
            int32_t v15 = XAie_StrmConnCctEnable(v1, XAie_TileLoc(1, 0), SOUTH, 7, NORTH, 1);
            int32_t v16 = XAie_StrmConnCctEnable(v1, XAie_TileLoc(1, 1), SOUTH, 1, NORTH, 1);
            int32_t v17 = XAie_StrmConnCctEnable(v1, XAie_TileLoc(1, 2), SOUTH, 1, NORTH, 1);
            int32_t v18 = XAie_StrmConnCctEnable(v1, XAie_TileLoc(1, 3), SOUTH, 1, EAST, 1);
            int32_t v19 = XAie_StrmConnCctEnable(v1, XAie_TileLoc(2, 3), WEST, 1, NORTH, 0);
            int32_t v20 = XAie_StrmConnCctEnable(v1, XAie_TileLoc(2, 3), WEST, 1, DMA, 1);
            int32_t v21 = XAie_StrmConnCctEnable(v1, XAie_TileLoc(2, 4), SOUTH, 0, NORTH, 0);
            int32_t v22 = XAie_StrmConnCctEnable(v1, XAie_TileLoc(2, 4), SOUTH, 0, DMA, 1);
            int32_t v23 = XAie_StrmConnCctEnable(v1, XAie_TileLoc(2, 5), SOUTH, 0, NORTH, 0);
            int32_t v24 = XAie_StrmConnCctEnable(v1, XAie_TileLoc(2, 5), SOUTH, 0, DMA, 1);
            int32_t v25 = XAie_StrmConnCctEnable(v1, XAie_TileLoc(2, 6), SOUTH, 0, DMA, 1);
        }

        // round is 2 hw split in : col -----------
        if (v2) {
            int32_t v26 = XAie_EnableShimDmaToAieStrmPort(v1, XAie_TileLoc(4, 0), 3);
            int32_t v27 = XAie_StrmConnCctEnable(v1, XAie_TileLoc(4, 0), SOUTH, 3, NORTH, 0);
            int32_t v28 = XAie_StrmConnCctEnable(v1, XAie_TileLoc(4, 1), SOUTH, 0, NORTH, 0);
            int32_t v29 = XAie_StrmConnCctEnable(v1, XAie_TileLoc(4, 2), SOUTH, 0, NORTH, 0);
            int32_t v30 = XAie_StrmConnCctEnable(v1, XAie_TileLoc(4, 3), SOUTH, 0, WEST, 1);
            int32_t v31 = XAie_StrmConnCctEnable(v1, XAie_TileLoc(3, 3), EAST, 1, NORTH, 0);
            int32_t v32 = XAie_StrmConnCctEnable(v1, XAie_TileLoc(3, 3), EAST, 1, DMA, 1);
            int32_t v33 = XAie_StrmConnCctEnable(v1, XAie_TileLoc(3, 4), SOUTH, 0, NORTH, 0);
            int32_t v34 = XAie_StrmConnCctEnable(v1, XAie_TileLoc(3, 4), SOUTH, 0, DMA, 1);
            int32_t v35 = XAie_StrmConnCctEnable(v1, XAie_TileLoc(3, 5), SOUTH, 0, NORTH, 0);
            int32_t v36 = XAie_StrmConnCctEnable(v1, XAie_TileLoc(3, 5), SOUTH, 0, DMA, 1);
            int32_t v37 = XAie_StrmConnCctEnable(v1, XAie_TileLoc(3, 6), SOUTH, 0, DMA, 1);
        }

        // round is 3 hw split in : col -----------
        if (v2) {
            int32_t v38 = XAie_EnableShimDmaToAieStrmPort(v1, XAie_TileLoc(4, 0), 7);
            int32_t v39 = XAie_StrmConnCctEnable(v1, XAie_TileLoc(4, 0), SOUTH, 7, NORTH, 1);
            int32_t v40 = XAie_StrmConnCctEnable(v1, XAie_TileLoc(4, 1), SOUTH, 1, NORTH, 1);
            int32_t v41 = XAie_StrmConnCctEnable(v1, XAie_TileLoc(4, 2), SOUTH, 1, NORTH, 1);
            int32_t v42 = XAie_StrmConnCctEnable(v1, XAie_TileLoc(4, 3), SOUTH, 1, NORTH, 0);
            int32_t v43 = XAie_StrmConnCctEnable(v1, XAie_TileLoc(4, 3), SOUTH, 1, DMA, 1);
            int32_t v44 = XAie_StrmConnCctEnable(v1, XAie_TileLoc(4, 4), SOUTH, 0, NORTH, 0);
            int32_t v45 = XAie_StrmConnCctEnable(v1, XAie_TileLoc(4, 4), SOUTH, 0, DMA, 1);
            int32_t v46 = XAie_StrmConnCctEnable(v1, XAie_TileLoc(4, 5), SOUTH, 0, NORTH, 0);
            int32_t v47 = XAie_StrmConnCctEnable(v1, XAie_TileLoc(4, 5), SOUTH, 0, DMA, 1);
            int32_t v48 = XAie_StrmConnCctEnable(v1, XAie_TileLoc(4, 6), SOUTH, 0, DMA, 1);
        }

    } //----routing creation in col ----end-------

    { //----routing creation in row ----start-------

        // round is 0 hw split in : row -----------
        if (v2) {
            int32_t v49 = XAie_EnableShimDmaToAieStrmPort(v1, XAie_TileLoc(5, 0), 3);
            int32_t v50 = XAie_StrmConnCctEnable(v1, XAie_TileLoc(5, 0), SOUTH, 3, NORTH, 0);
            int32_t v51 = XAie_StrmConnCctEnable(v1, XAie_TileLoc(5, 1), SOUTH, 0, NORTH, 0);
            int32_t v52 = XAie_StrmConnCctEnable(v1, XAie_TileLoc(5, 2), SOUTH, 0, NORTH, 0);
            int32_t v53 = XAie_StrmConnCctEnable(v1, XAie_TileLoc(5, 3), SOUTH, 0, WEST, 1);
            int32_t v54 = XAie_StrmConnCctEnable(v1, XAie_TileLoc(4, 3), EAST, 1, WEST, 2);
            int32_t v55 = XAie_StrmConnCctEnable(v1, XAie_TileLoc(4, 3), EAST, 1, DMA, 0);
            int32_t v56 = XAie_StrmConnCctEnable(v1, XAie_TileLoc(3, 3), EAST, 2, WEST, 1);
            int32_t v57 = XAie_StrmConnCctEnable(v1, XAie_TileLoc(3, 3), EAST, 2, DMA, 0);
            int32_t v58 = XAie_StrmConnCctEnable(v1, XAie_TileLoc(2, 3), EAST, 1, WEST, 2);
            int32_t v59 = XAie_StrmConnCctEnable(v1, XAie_TileLoc(2, 3), EAST, 1, DMA, 0);
            int32_t v60 = XAie_StrmConnCctEnable(v1, XAie_TileLoc(1, 3), EAST, 2, DMA, 0);
            int32_t v61 =
                XAie_StrmPktSwSlaveSlotEnable(v1, XAie_TileLoc(1, 3), DMA, 0, 0, XAie_PacketInit(4, 0), 31, 0, 2);
            int32_t v62 = XAie_StrmPktSwSlavePortEnable(v1, XAie_TileLoc(1, 3), DMA, 0);
            int32_t v63 =
                XAie_StrmPktSwMstrPortEnable(v1, XAie_TileLoc(1, 3), EAST, 3, XAIE_SS_PKT_DONOT_DROP_HEADER, 2, 1);
            int32_t v64 =
                XAie_StrmPktSwSlaveSlotEnable(v1, XAie_TileLoc(2, 3), WEST, 3, 3, XAie_PacketInit(0, 0), 0, 0, 2);
            int32_t v65 = XAie_StrmPktSwSlavePortEnable(v1, XAie_TileLoc(2, 3), WEST, 3);
            int32_t v66 =
                XAie_StrmPktSwSlaveSlotEnable(v1, XAie_TileLoc(2, 3), DMA, 0, 0, XAie_PacketInit(5, 0), 31, 0, 2);
            int32_t v67 = XAie_StrmPktSwSlavePortEnable(v1, XAie_TileLoc(2, 3), DMA, 0);
            int32_t v68 =
                XAie_StrmPktSwMstrPortEnable(v1, XAie_TileLoc(2, 3), EAST, 2, XAIE_SS_PKT_DONOT_DROP_HEADER, 2, 1);
            int32_t v69 =
                XAie_StrmPktSwSlaveSlotEnable(v1, XAie_TileLoc(3, 3), WEST, 2, 3, XAie_PacketInit(0, 0), 0, 0, 2);
            int32_t v70 = XAie_StrmPktSwSlavePortEnable(v1, XAie_TileLoc(3, 3), WEST, 2);
            int32_t v71 =
                XAie_StrmPktSwSlaveSlotEnable(v1, XAie_TileLoc(3, 3), DMA, 0, 0, XAie_PacketInit(6, 0), 31, 0, 2);
            int32_t v72 = XAie_StrmPktSwSlavePortEnable(v1, XAie_TileLoc(3, 3), DMA, 0);
            int32_t v73 =
                XAie_StrmPktSwMstrPortEnable(v1, XAie_TileLoc(3, 3), EAST, 3, XAIE_SS_PKT_DONOT_DROP_HEADER, 2, 1);
            int32_t v74 =
                XAie_StrmPktSwSlaveSlotEnable(v1, XAie_TileLoc(4, 3), WEST, 3, 3, XAie_PacketInit(0, 0), 0, 0, 2);
            int32_t v75 = XAie_StrmPktSwSlavePortEnable(v1, XAie_TileLoc(4, 3), WEST, 3);
            int32_t v76 =
                XAie_StrmPktSwSlaveSlotEnable(v1, XAie_TileLoc(4, 3), DMA, 0, 0, XAie_PacketInit(7, 0), 31, 0, 2);
            int32_t v77 = XAie_StrmPktSwSlavePortEnable(v1, XAie_TileLoc(4, 3), DMA, 0);
            int32_t v78 = XAie_EnableAieToShimDmaStrmPort(v1, XAie_TileLoc(4, 0), 2);
            int32_t v79 =
                XAie_StrmPktSwMstrPortEnable(v1, XAie_TileLoc(4, 3), SOUTH, 2, XAIE_SS_PKT_DONOT_DROP_HEADER, 2, 1);
            int32_t v80 = XAie_StrmConnCctEnable(v1, XAie_TileLoc(4, 2), NORTH, 2, SOUTH, 2);
            int32_t v81 = XAie_StrmConnCctEnable(v1, XAie_TileLoc(4, 1), NORTH, 2, SOUTH, 2);
            int32_t v82 = XAie_StrmConnCctEnable(v1, XAie_TileLoc(4, 0), NORTH, 2, SOUTH, 2);
        }

        // round is 1 hw split in : row -----------
        if (v2) {
            int32_t v83 = XAie_EnableShimDmaToAieStrmPort(v1, XAie_TileLoc(5, 0), 7);
            int32_t v84 = XAie_StrmConnCctEnable(v1, XAie_TileLoc(5, 0), SOUTH, 7, NORTH, 1);
            int32_t v85 = XAie_StrmConnCctEnable(v1, XAie_TileLoc(5, 1), SOUTH, 1, NORTH, 1);
            int32_t v86 = XAie_StrmConnCctEnable(v1, XAie_TileLoc(5, 2), SOUTH, 1, NORTH, 1);
            int32_t v87 = XAie_StrmConnCctEnable(v1, XAie_TileLoc(5, 3), SOUTH, 1, WEST, 2);
            int32_t v88 = XAie_StrmConnCctEnable(v1, XAie_TileLoc(4, 3), EAST, 2, NORTH, 1);
            int32_t v89 = XAie_StrmConnCctEnable(v1, XAie_TileLoc(4, 4), SOUTH, 1, WEST, 1);
            int32_t v90 = XAie_StrmConnCctEnable(v1, XAie_TileLoc(4, 4), SOUTH, 1, DMA, 0);
            int32_t v91 = XAie_StrmConnCctEnable(v1, XAie_TileLoc(3, 4), EAST, 1, WEST, 1);
            int32_t v92 = XAie_StrmConnCctEnable(v1, XAie_TileLoc(3, 4), EAST, 1, DMA, 0);
            int32_t v93 = XAie_StrmConnCctEnable(v1, XAie_TileLoc(2, 4), EAST, 1, WEST, 1);
            int32_t v94 = XAie_StrmConnCctEnable(v1, XAie_TileLoc(2, 4), EAST, 1, DMA, 0);
            int32_t v95 = XAie_StrmConnCctEnable(v1, XAie_TileLoc(1, 4), EAST, 1, DMA, 0);
            int32_t v96 =
                XAie_StrmPktSwSlaveSlotEnable(v1, XAie_TileLoc(1, 4), DMA, 0, 0, XAie_PacketInit(4, 0), 31, 0, 2);
            int32_t v97 = XAie_StrmPktSwSlavePortEnable(v1, XAie_TileLoc(1, 4), DMA, 0);
            int32_t v98 =
                XAie_StrmPktSwMstrPortEnable(v1, XAie_TileLoc(1, 4), EAST, 2, XAIE_SS_PKT_DONOT_DROP_HEADER, 2, 1);
            int32_t v99 =
                XAie_StrmPktSwSlaveSlotEnable(v1, XAie_TileLoc(2, 4), WEST, 2, 3, XAie_PacketInit(0, 0), 0, 0, 2);
            int32_t v100 = XAie_StrmPktSwSlavePortEnable(v1, XAie_TileLoc(2, 4), WEST, 2);
            int32_t v101 =
                XAie_StrmPktSwSlaveSlotEnable(v1, XAie_TileLoc(2, 4), DMA, 0, 0, XAie_PacketInit(5, 0), 31, 0, 2);
            int32_t v102 = XAie_StrmPktSwSlavePortEnable(v1, XAie_TileLoc(2, 4), DMA, 0);
            int32_t v103 =
                XAie_StrmPktSwMstrPortEnable(v1, XAie_TileLoc(2, 4), EAST, 2, XAIE_SS_PKT_DONOT_DROP_HEADER, 2, 1);
            int32_t v104 =
                XAie_StrmPktSwSlaveSlotEnable(v1, XAie_TileLoc(3, 4), WEST, 2, 3, XAie_PacketInit(0, 0), 0, 0, 2);
            int32_t v105 = XAie_StrmPktSwSlavePortEnable(v1, XAie_TileLoc(3, 4), WEST, 2);
            int32_t v106 =
                XAie_StrmPktSwSlaveSlotEnable(v1, XAie_TileLoc(3, 4), DMA, 0, 0, XAie_PacketInit(6, 0), 31, 0, 2);
            int32_t v107 = XAie_StrmPktSwSlavePortEnable(v1, XAie_TileLoc(3, 4), DMA, 0);
            int32_t v108 =
                XAie_StrmPktSwMstrPortEnable(v1, XAie_TileLoc(3, 4), EAST, 2, XAIE_SS_PKT_DONOT_DROP_HEADER, 2, 1);
            int32_t v109 =
                XAie_StrmPktSwSlaveSlotEnable(v1, XAie_TileLoc(4, 4), WEST, 2, 3, XAie_PacketInit(0, 0), 0, 0, 2);
            int32_t v110 = XAie_StrmPktSwSlavePortEnable(v1, XAie_TileLoc(4, 4), WEST, 2);
            int32_t v111 =
                XAie_StrmPktSwSlaveSlotEnable(v1, XAie_TileLoc(4, 4), DMA, 0, 0, XAie_PacketInit(7, 0), 31, 0, 2);
            int32_t v112 = XAie_StrmPktSwSlavePortEnable(v1, XAie_TileLoc(4, 4), DMA, 0);
            int32_t v113 = XAie_EnableAieToShimDmaStrmPort(v1, XAie_TileLoc(4, 0), 3);
            int32_t v114 =
                XAie_StrmPktSwMstrPortEnable(v1, XAie_TileLoc(4, 4), SOUTH, 2, XAIE_SS_PKT_DONOT_DROP_HEADER, 2, 1);
            int32_t v115 = XAie_StrmConnCctEnable(v1, XAie_TileLoc(4, 3), NORTH, 2, SOUTH, 3);
            int32_t v116 = XAie_StrmConnCctEnable(v1, XAie_TileLoc(4, 2), NORTH, 3, SOUTH, 3);
            int32_t v117 = XAie_StrmConnCctEnable(v1, XAie_TileLoc(4, 1), NORTH, 3, SOUTH, 3);
            int32_t v118 = XAie_StrmConnCctEnable(v1, XAie_TileLoc(4, 0), NORTH, 3, SOUTH, 3);
        }

        // round is 2 hw split in : row -----------
        if (v2) {
            int32_t v119 = XAie_EnableShimDmaToAieStrmPort(v1, XAie_TileLoc(12, 0), 3);
            int32_t v120 = XAie_StrmConnCctEnable(v1, XAie_TileLoc(12, 0), SOUTH, 3, NORTH, 0);
            int32_t v121 = XAie_StrmConnCctEnable(v1, XAie_TileLoc(12, 1), SOUTH, 0, NORTH, 0);
            int32_t v122 = XAie_StrmConnCctEnable(v1, XAie_TileLoc(12, 2), SOUTH, 0, NORTH, 0);
            int32_t v123 = XAie_StrmConnCctEnable(v1, XAie_TileLoc(12, 3), SOUTH, 0, WEST, 1);
            int32_t v124 = XAie_StrmConnCctEnable(v1, XAie_TileLoc(11, 3), EAST, 1, WEST, 1);
            int32_t v125 = XAie_StrmConnCctEnable(v1, XAie_TileLoc(10, 3), EAST, 1, WEST, 1);
            int32_t v126 = XAie_StrmConnCctEnable(v1, XAie_TileLoc(9, 3), EAST, 1, WEST, 1);
            int32_t v127 = XAie_StrmConnCctEnable(v1, XAie_TileLoc(8, 3), EAST, 1, WEST, 1);
            int32_t v128 = XAie_StrmConnCctEnable(v1, XAie_TileLoc(7, 3), EAST, 1, WEST, 1);
            int32_t v129 = XAie_StrmConnCctEnable(v1, XAie_TileLoc(6, 3), EAST, 1, WEST, 1);
            int32_t v130 = XAie_StrmConnCctEnable(v1, XAie_TileLoc(5, 3), EAST, 1, WEST, 3);
            int32_t v131 = XAie_StrmConnCctEnable(v1, XAie_TileLoc(4, 3), EAST, 3, NORTH, 3);
            int32_t v132 = XAie_StrmConnCctEnable(v1, XAie_TileLoc(4, 4), SOUTH, 3, WEST, 3);
            int32_t v133 = XAie_StrmConnCctEnable(v1, XAie_TileLoc(4, 4), SOUTH, 3, NORTH, 1);
            int32_t v134 = XAie_StrmConnCctEnable(v1, XAie_TileLoc(3, 4), EAST, 3, WEST, 3);
            int32_t v135 = XAie_StrmConnCctEnable(v1, XAie_TileLoc(3, 4), EAST, 3, NORTH, 1);
            int32_t v136 = XAie_StrmConnCctEnable(v1, XAie_TileLoc(2, 4), EAST, 3, WEST, 3);
            int32_t v137 = XAie_StrmConnCctEnable(v1, XAie_TileLoc(2, 4), EAST, 3, NORTH, 1);
            int32_t v138 = XAie_StrmConnCctEnable(v1, XAie_TileLoc(1, 4), EAST, 3, NORTH, 1);
            int32_t v139 = XAie_StrmConnCctEnable(v1, XAie_TileLoc(1, 5), SOUTH, 1, DMA, 0);
            int32_t v140 = XAie_StrmConnCctEnable(v1, XAie_TileLoc(2, 5), SOUTH, 1, DMA, 0);
            int32_t v141 = XAie_StrmConnCctEnable(v1, XAie_TileLoc(3, 5), SOUTH, 1, DMA, 0);
            int32_t v142 = XAie_StrmConnCctEnable(v1, XAie_TileLoc(4, 5), SOUTH, 1, DMA, 0);
            int32_t v143 =
                XAie_StrmPktSwSlaveSlotEnable(v1, XAie_TileLoc(1, 5), DMA, 0, 0, XAie_PacketInit(4, 0), 31, 0, 2);
            int32_t v144 = XAie_StrmPktSwSlavePortEnable(v1, XAie_TileLoc(1, 5), DMA, 0);
            int32_t v145 =
                XAie_StrmPktSwMstrPortEnable(v1, XAie_TileLoc(1, 5), EAST, 1, XAIE_SS_PKT_DONOT_DROP_HEADER, 2, 1);
            int32_t v146 =
                XAie_StrmPktSwSlaveSlotEnable(v1, XAie_TileLoc(2, 5), WEST, 1, 3, XAie_PacketInit(0, 0), 0, 0, 2);
            int32_t v147 = XAie_StrmPktSwSlavePortEnable(v1, XAie_TileLoc(2, 5), WEST, 1);
            int32_t v148 =
                XAie_StrmPktSwSlaveSlotEnable(v1, XAie_TileLoc(2, 5), DMA, 0, 0, XAie_PacketInit(5, 0), 31, 0, 2);
            int32_t v149 = XAie_StrmPktSwSlavePortEnable(v1, XAie_TileLoc(2, 5), DMA, 0);
            int32_t v150 =
                XAie_StrmPktSwMstrPortEnable(v1, XAie_TileLoc(2, 5), EAST, 1, XAIE_SS_PKT_DONOT_DROP_HEADER, 2, 1);
            int32_t v151 =
                XAie_StrmPktSwSlaveSlotEnable(v1, XAie_TileLoc(3, 5), WEST, 1, 3, XAie_PacketInit(0, 0), 0, 0, 2);
            int32_t v152 = XAie_StrmPktSwSlavePortEnable(v1, XAie_TileLoc(3, 5), WEST, 1);
            int32_t v153 =
                XAie_StrmPktSwSlaveSlotEnable(v1, XAie_TileLoc(3, 5), DMA, 0, 0, XAie_PacketInit(6, 0), 31, 0, 2);
            int32_t v154 = XAie_StrmPktSwSlavePortEnable(v1, XAie_TileLoc(3, 5), DMA, 0);
            int32_t v155 =
                XAie_StrmPktSwMstrPortEnable(v1, XAie_TileLoc(3, 5), EAST, 1, XAIE_SS_PKT_DONOT_DROP_HEADER, 2, 1);
            int32_t v156 =
                XAie_StrmPktSwSlaveSlotEnable(v1, XAie_TileLoc(4, 5), WEST, 1, 3, XAie_PacketInit(0, 0), 0, 0, 2);
            int32_t v157 = XAie_StrmPktSwSlavePortEnable(v1, XAie_TileLoc(4, 5), WEST, 1);
            int32_t v158 =
                XAie_StrmPktSwSlaveSlotEnable(v1, XAie_TileLoc(4, 5), DMA, 0, 0, XAie_PacketInit(7, 0), 31, 0, 2);
            int32_t v159 = XAie_StrmPktSwSlavePortEnable(v1, XAie_TileLoc(4, 5), DMA, 0);
            int32_t v160 = XAie_EnableAieToShimDmaStrmPort(v1, XAie_TileLoc(5, 0), 2);
            int32_t v161 =
                XAie_StrmPktSwMstrPortEnable(v1, XAie_TileLoc(4, 5), SOUTH, 2, XAIE_SS_PKT_DONOT_DROP_HEADER, 2, 1);
            int32_t v162 = XAie_StrmConnCctEnable(v1, XAie_TileLoc(4, 4), NORTH, 2, EAST, 1);
            int32_t v163 = XAie_StrmConnCctEnable(v1, XAie_TileLoc(5, 4), WEST, 1, SOUTH, 0);
            int32_t v164 = XAie_StrmConnCctEnable(v1, XAie_TileLoc(5, 3), NORTH, 0, SOUTH, 2);
            int32_t v165 = XAie_StrmConnCctEnable(v1, XAie_TileLoc(5, 2), NORTH, 2, SOUTH, 2);
            int32_t v166 = XAie_StrmConnCctEnable(v1, XAie_TileLoc(5, 1), NORTH, 2, SOUTH, 2);
            int32_t v167 = XAie_StrmConnCctEnable(v1, XAie_TileLoc(5, 0), NORTH, 2, SOUTH, 2);
        }

        // round is 3 hw split in : row -----------
        if (v2) {
            int32_t v168 = XAie_EnableShimDmaToAieStrmPort(v1, XAie_TileLoc(12, 0), 7);
            int32_t v169 = XAie_StrmConnCctEnable(v1, XAie_TileLoc(12, 0), SOUTH, 7, NORTH, 1);
            int32_t v170 = XAie_StrmConnCctEnable(v1, XAie_TileLoc(12, 1), SOUTH, 1, NORTH, 1);
            int32_t v171 = XAie_StrmConnCctEnable(v1, XAie_TileLoc(12, 2), SOUTH, 1, NORTH, 1);
            int32_t v172 = XAie_StrmConnCctEnable(v1, XAie_TileLoc(12, 3), SOUTH, 1, WEST, 2);
            int32_t v173 = XAie_StrmConnCctEnable(v1, XAie_TileLoc(11, 3), EAST, 2, WEST, 2);
            int32_t v174 = XAie_StrmConnCctEnable(v1, XAie_TileLoc(10, 3), EAST, 2, WEST, 2);
            int32_t v175 = XAie_StrmConnCctEnable(v1, XAie_TileLoc(9, 3), EAST, 2, WEST, 2);
            int32_t v176 = XAie_StrmConnCctEnable(v1, XAie_TileLoc(8, 3), EAST, 2, WEST, 2);
            int32_t v177 = XAie_StrmConnCctEnable(v1, XAie_TileLoc(7, 3), EAST, 2, WEST, 2);
            int32_t v178 = XAie_StrmConnCctEnable(v1, XAie_TileLoc(6, 3), EAST, 2, WEST, 2);
            int32_t v179 = XAie_StrmConnCctEnable(v1, XAie_TileLoc(5, 3), EAST, 2, NORTH, 1);
            int32_t v180 = XAie_StrmConnCctEnable(v1, XAie_TileLoc(5, 4), SOUTH, 1, WEST, 2);
            int32_t v181 = XAie_StrmConnCctEnable(v1, XAie_TileLoc(4, 4), EAST, 2, NORTH, 3);
            int32_t v182 = XAie_StrmConnCctEnable(v1, XAie_TileLoc(4, 5), SOUTH, 3, WEST, 2);
            int32_t v183 = XAie_StrmConnCctEnable(v1, XAie_TileLoc(4, 5), SOUTH, 3, NORTH, 1);
            int32_t v184 = XAie_StrmConnCctEnable(v1, XAie_TileLoc(3, 5), EAST, 2, WEST, 2);
            int32_t v185 = XAie_StrmConnCctEnable(v1, XAie_TileLoc(3, 5), EAST, 2, NORTH, 1);
            int32_t v186 = XAie_StrmConnCctEnable(v1, XAie_TileLoc(2, 5), EAST, 2, WEST, 2);
            int32_t v187 = XAie_StrmConnCctEnable(v1, XAie_TileLoc(2, 5), EAST, 2, NORTH, 1);
            int32_t v188 = XAie_StrmConnCctEnable(v1, XAie_TileLoc(1, 5), EAST, 2, NORTH, 1);
            int32_t v189 = XAie_StrmConnCctEnable(v1, XAie_TileLoc(1, 6), SOUTH, 1, DMA, 0);
            int32_t v190 = XAie_StrmConnCctEnable(v1, XAie_TileLoc(2, 6), SOUTH, 1, DMA, 0);
            int32_t v191 = XAie_StrmConnCctEnable(v1, XAie_TileLoc(3, 6), SOUTH, 1, DMA, 0);
            int32_t v192 = XAie_StrmConnCctEnable(v1, XAie_TileLoc(4, 6), SOUTH, 1, DMA, 0);
            int32_t v193 =
                XAie_StrmPktSwSlaveSlotEnable(v1, XAie_TileLoc(1, 6), DMA, 0, 0, XAie_PacketInit(4, 0), 31, 0, 2);
            int32_t v194 = XAie_StrmPktSwSlavePortEnable(v1, XAie_TileLoc(1, 6), DMA, 0);
            int32_t v195 =
                XAie_StrmPktSwMstrPortEnable(v1, XAie_TileLoc(1, 6), EAST, 1, XAIE_SS_PKT_DONOT_DROP_HEADER, 2, 1);
            int32_t v196 =
                XAie_StrmPktSwSlaveSlotEnable(v1, XAie_TileLoc(2, 6), WEST, 1, 3, XAie_PacketInit(0, 0), 0, 0, 2);
            int32_t v197 = XAie_StrmPktSwSlavePortEnable(v1, XAie_TileLoc(2, 6), WEST, 1);
            int32_t v198 =
                XAie_StrmPktSwSlaveSlotEnable(v1, XAie_TileLoc(2, 6), DMA, 0, 0, XAie_PacketInit(5, 0), 31, 0, 2);
            int32_t v199 = XAie_StrmPktSwSlavePortEnable(v1, XAie_TileLoc(2, 6), DMA, 0);
            int32_t v200 =
                XAie_StrmPktSwMstrPortEnable(v1, XAie_TileLoc(2, 6), EAST, 1, XAIE_SS_PKT_DONOT_DROP_HEADER, 2, 1);
            int32_t v201 =
                XAie_StrmPktSwSlaveSlotEnable(v1, XAie_TileLoc(3, 6), WEST, 1, 3, XAie_PacketInit(0, 0), 0, 0, 2);
            int32_t v202 = XAie_StrmPktSwSlavePortEnable(v1, XAie_TileLoc(3, 6), WEST, 1);
            int32_t v203 =
                XAie_StrmPktSwSlaveSlotEnable(v1, XAie_TileLoc(3, 6), DMA, 0, 0, XAie_PacketInit(6, 0), 31, 0, 2);
            int32_t v204 = XAie_StrmPktSwSlavePortEnable(v1, XAie_TileLoc(3, 6), DMA, 0);
            int32_t v205 =
                XAie_StrmPktSwMstrPortEnable(v1, XAie_TileLoc(3, 6), EAST, 1, XAIE_SS_PKT_DONOT_DROP_HEADER, 2, 1);
            int32_t v206 =
                XAie_StrmPktSwSlaveSlotEnable(v1, XAie_TileLoc(4, 6), WEST, 1, 3, XAie_PacketInit(0, 0), 0, 0, 2);
            int32_t v207 = XAie_StrmPktSwSlavePortEnable(v1, XAie_TileLoc(4, 6), WEST, 1);
            int32_t v208 =
                XAie_StrmPktSwSlaveSlotEnable(v1, XAie_TileLoc(4, 6), DMA, 0, 0, XAie_PacketInit(7, 0), 31, 0, 2);
            int32_t v209 = XAie_StrmPktSwSlavePortEnable(v1, XAie_TileLoc(4, 6), DMA, 0);
            int32_t v210 = XAie_EnableAieToShimDmaStrmPort(v1, XAie_TileLoc(5, 0), 3);
            int32_t v211 =
                XAie_StrmPktSwMstrPortEnable(v1, XAie_TileLoc(4, 6), SOUTH, 2, XAIE_SS_PKT_DONOT_DROP_HEADER, 2, 1);
            int32_t v212 = XAie_StrmConnCctEnable(v1, XAie_TileLoc(4, 5), NORTH, 2, EAST, 1);
            int32_t v213 = XAie_StrmConnCctEnable(v1, XAie_TileLoc(5, 5), WEST, 1, SOUTH, 0);
            int32_t v214 = XAie_StrmConnCctEnable(v1, XAie_TileLoc(5, 4), NORTH, 0, SOUTH, 2);
            int32_t v215 = XAie_StrmConnCctEnable(v1, XAie_TileLoc(5, 3), NORTH, 2, SOUTH, 3);
            int32_t v216 = XAie_StrmConnCctEnable(v1, XAie_TileLoc(5, 2), NORTH, 3, SOUTH, 3);
            int32_t v217 = XAie_StrmConnCctEnable(v1, XAie_TileLoc(5, 1), NORTH, 3, SOUTH, 3);
            int32_t v218 = XAie_StrmConnCctEnable(v1, XAie_TileLoc(5, 0), NORTH, 3, SOUTH, 3);
        }

    } //----routing creation in row ----end-------

    return;
}
#include <xaiengine.h>
