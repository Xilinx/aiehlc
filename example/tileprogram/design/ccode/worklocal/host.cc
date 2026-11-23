/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/
#include "aie_runtime.h"
#include "aie_runtime_debug.h"
void host_canonicalized(void *v1, void *v2, void *v3) {
    void *v4 = __runtime_buffer_offset(v1, 0);
    XAie_LocType v5 = XAie_TileLoc(2, 0);
    /* DMA BD Config: bd_id=0, offset=0, len=32, enable_packet=false, packet_id=0, next_bd=-1, acquire_lock_id=0,
     * acquire_lock_val=0, release_lock_id=0, release_lock_val=0 */
    void *v6 = __runtime_buffer_arg(v4);
    XAie_DmaDesc v7 = __Runtime_dma_bd_config(g_DevInst, v5, v6, 0, 32, -1, 0, 0, 0, 0, 0, 0);
    /* Create IO: channel_id=0, bd_id=0, tile=(2,0), direction=MM2S */
    io v8 = __Runtime_dma_createio_4(v5, v7, 0, 0, DMA_MM2S);
    XAie_LocType v9 = XAie_TileLoc(0, 3);
    /* DMA BD Config: bd_id=1, offset=0, len=16, enable_packet=false, packet_id=0, next_bd=0, acquire_lock_id=0,
     * acquire_lock_val=-1, release_lock_id=1, release_lock_val=1 */
    void *v10 = __runtime_buffer_arg((void *)32832);
    XAie_DmaDesc v11 = __Runtime_dma_bd_config(g_DevInst, v9, v10, 1, 16, 0, 0, 0, 0, -1, 1, 1);
    /* Lock init: tile(0,3) lock=0 init_value=2 */
    XAie_LockSetValue(g_DevInst, XAie_TileLoc(0, 3), XAie_LockInit(0, 2));
    /* DMA BD Config: bd_id=0, offset=0, len=16, enable_packet=false, packet_id=0, next_bd=1, acquire_lock_id=0,
     * acquire_lock_val=-1, release_lock_id=1, release_lock_val=1 */
    void *v12 = __runtime_buffer_arg((void *)32768);
    XAie_DmaDesc v13 = __Runtime_dma_bd_config(g_DevInst, v9, v12, 0, 16, 1, 0, 0, 0, -1, 1, 1);
    /* Create IO: channel_id=0, bd_id=0, tile=(0,3), direction=S2MM */
    io v14 = __Runtime_dma_createio_4(v9, v13, 0, 0, DMA_S2MM);
    /* Allocated BD ID 0 for tile (0,3) */
    XAie_LocType v15 = XAie_TileLoc(1, 3);
    /* DMA BD Config: bd_id=1, offset=0, len=16, enable_packet=false, packet_id=0, next_bd=0, acquire_lock_id=0,
     * acquire_lock_val=-1, release_lock_id=1, release_lock_val=1 */
    void *v16 = __runtime_buffer_arg((void *)32832);
    XAie_DmaDesc v17 = __Runtime_dma_bd_config(g_DevInst, v15, v16, 1, 16, 0, 0, 0, 0, -1, 1, 1);
    /* Lock init: tile(1,3) lock=0 init_value=2 */
    XAie_LockSetValue(g_DevInst, XAie_TileLoc(1, 3), XAie_LockInit(0, 2));
    /* DMA BD Config: bd_id=0, offset=0, len=16, enable_packet=false, packet_id=0, next_bd=1, acquire_lock_id=0,
     * acquire_lock_val=-1, release_lock_id=1, release_lock_val=1 */
    void *v18 = __runtime_buffer_arg((void *)32768);
    XAie_DmaDesc v19 = __Runtime_dma_bd_config(g_DevInst, v15, v18, 0, 16, 1, 0, 0, 0, -1, 1, 1);
    /* Create IO: channel_id=0, bd_id=0, tile=(1,3), direction=S2MM */
    io v20 = __Runtime_dma_createio_4(v15, v19, 0, 0, DMA_S2MM);
    /* Allocated BD ID 0 for tile (1,3) */
    /* Allocated BD ID 0 for tile (2,0) */
    void *v21 = __runtime_buffer_offset(v1, 128);
    /* DMA BD Config: bd_id=1, offset=0, len=32, enable_packet=false, packet_id=0, next_bd=-1, acquire_lock_id=0,
     * acquire_lock_val=0, release_lock_id=0, release_lock_val=0 */
    void *v22 = __runtime_buffer_arg(v21);
    XAie_DmaDesc v23 = __Runtime_dma_bd_config(g_DevInst, v5, v22, 1, 32, -1, 0, 0, 0, 0, 0, 0);
    /* Create IO: channel_id=1, bd_id=1, tile=(2,0), direction=MM2S */
    io v24 = __Runtime_dma_createio_4(v5, v23, 1, 1, DMA_MM2S);
    XAie_LocType v25 = XAie_TileLoc(0, 4);
    void *v26 = __runtime_buffer_offset(v21, 128);
    /* DMA BD Config: bd_id=1, offset=0, len=16, enable_packet=false, packet_id=0, next_bd=0, acquire_lock_id=0,
     * acquire_lock_val=-1, release_lock_id=1, release_lock_val=1 */
    void *v27 = __runtime_buffer_arg((void *)32832);
    XAie_DmaDesc v28 = __Runtime_dma_bd_config(g_DevInst, v25, v27, 1, 16, 0, 0, 0, 0, -1, 1, 1);
    /* Lock init: tile(0,4) lock=0 init_value=2 */
    XAie_LockSetValue(g_DevInst, XAie_TileLoc(0, 4), XAie_LockInit(0, 2));
    /* DMA BD Config: bd_id=0, offset=0, len=16, enable_packet=false, packet_id=0, next_bd=1, acquire_lock_id=0,
     * acquire_lock_val=-1, release_lock_id=1, release_lock_val=1 */
    void *v29 = __runtime_buffer_arg((void *)32768);
    XAie_DmaDesc v30 = __Runtime_dma_bd_config(g_DevInst, v25, v29, 0, 16, 1, 0, 0, 0, -1, 1, 1);
    /* Create IO: channel_id=0, bd_id=0, tile=(0,4), direction=S2MM */
    io v31 = __Runtime_dma_createio_4(v25, v30, 0, 0, DMA_S2MM);
    /* Allocated BD ID 0 for tile (0,4) */
    XAie_LocType v32 = XAie_TileLoc(1, 4);
    void *v33 = __runtime_buffer_offset(v21, 128);
    /* DMA BD Config: bd_id=1, offset=0, len=16, enable_packet=false, packet_id=0, next_bd=0, acquire_lock_id=0,
     * acquire_lock_val=-1, release_lock_id=1, release_lock_val=1 */
    void *v34 = __runtime_buffer_arg((void *)32832);
    XAie_DmaDesc v35 = __Runtime_dma_bd_config(g_DevInst, v32, v34, 1, 16, 0, 0, 0, 0, -1, 1, 1);
    /* Lock init: tile(1,4) lock=0 init_value=2 */
    XAie_LockSetValue(g_DevInst, XAie_TileLoc(1, 4), XAie_LockInit(0, 2));
    /* DMA BD Config: bd_id=0, offset=0, len=16, enable_packet=false, packet_id=0, next_bd=1, acquire_lock_id=0,
     * acquire_lock_val=-1, release_lock_id=1, release_lock_val=1 */
    void *v36 = __runtime_buffer_arg((void *)32768);
    XAie_DmaDesc v37 = __Runtime_dma_bd_config(g_DevInst, v32, v36, 0, 16, 1, 0, 0, 0, -1, 1, 1);
    /* Create IO: channel_id=0, bd_id=0, tile=(1,4), direction=S2MM */
    io v38 = __Runtime_dma_createio_4(v32, v37, 0, 0, DMA_S2MM);
    /* Allocated BD ID 0 for tile (1,4) */
    /* Allocated BD ID 1 for tile (2,0) */
    void *v39 = __runtime_buffer_offset(v2, 0);
    XAie_LocType v40 = XAie_TileLoc(3, 0);
    /* DMA BD Config: bd_id=0, offset=0, len=32, enable_packet=false, packet_id=0, next_bd=-1, acquire_lock_id=0,
     * acquire_lock_val=0, release_lock_id=0, release_lock_val=0 */
    void *v41 = __runtime_buffer_arg(v39);
    XAie_DmaDesc v42 = __Runtime_dma_bd_config(g_DevInst, v40, v41, 0, 32, -1, 0, 0, 0, 0, 0, 0);
    /* Create IO: channel_id=0, bd_id=0, tile=(3,0), direction=MM2S */
    io v43 = __Runtime_dma_createio_4(v40, v42, 0, 0, DMA_MM2S);
    /* DMA BD Config: bd_id=3, offset=0, len=16, enable_packet=false, packet_id=0, next_bd=2, acquire_lock_id=2,
     * acquire_lock_val=-1, release_lock_id=3, release_lock_val=1 */
    void *v44 = __runtime_buffer_arg((void *)32960);
    XAie_DmaDesc v45 = __Runtime_dma_bd_config(g_DevInst, v9, v44, 3, 16, 2, 0, 0, 2, -1, 3, 1);
    /* Lock init: tile(0,3) lock=2 init_value=2 */
    XAie_LockSetValue(g_DevInst, XAie_TileLoc(0, 3), XAie_LockInit(2, 2));
    /* DMA BD Config: bd_id=2, offset=0, len=16, enable_packet=false, packet_id=0, next_bd=3, acquire_lock_id=2,
     * acquire_lock_val=-1, release_lock_id=3, release_lock_val=1 */
    void *v46 = __runtime_buffer_arg((void *)32896);
    XAie_DmaDesc v47 = __Runtime_dma_bd_config(g_DevInst, v9, v46, 2, 16, 3, 0, 0, 2, -1, 3, 1);
    /* Create IO: channel_id=1, bd_id=2, tile=(0,3), direction=S2MM */
    io v48 = __Runtime_dma_createio_4(v9, v47, 1, 2, DMA_S2MM);
    /* Allocated BD ID 1 for tile (0,3) */
    /* DMA BD Config: bd_id=3, offset=0, len=16, enable_packet=false, packet_id=0, next_bd=2, acquire_lock_id=2,
     * acquire_lock_val=-1, release_lock_id=3, release_lock_val=1 */
    void *v49 = __runtime_buffer_arg((void *)32960);
    XAie_DmaDesc v50 = __Runtime_dma_bd_config(g_DevInst, v15, v49, 3, 16, 2, 0, 0, 2, -1, 3, 1);
    /* Lock init: tile(1,3) lock=2 init_value=2 */
    XAie_LockSetValue(g_DevInst, XAie_TileLoc(1, 3), XAie_LockInit(2, 2));
    /* DMA BD Config: bd_id=2, offset=0, len=16, enable_packet=false, packet_id=0, next_bd=3, acquire_lock_id=2,
     * acquire_lock_val=-1, release_lock_id=3, release_lock_val=1 */
    void *v51 = __runtime_buffer_arg((void *)32896);
    XAie_DmaDesc v52 = __Runtime_dma_bd_config(g_DevInst, v15, v51, 2, 16, 3, 0, 0, 2, -1, 3, 1);
    /* Create IO: channel_id=1, bd_id=2, tile=(1,3), direction=S2MM */
    io v53 = __Runtime_dma_createio_4(v15, v52, 1, 2, DMA_S2MM);
    /* Allocated BD ID 1 for tile (1,3) */
    /* Allocated BD ID 0 for tile (3,0) */
    void *v54 = __runtime_buffer_offset(v2, 128);
    /* DMA BD Config: bd_id=1, offset=0, len=32, enable_packet=false, packet_id=0, next_bd=-1, acquire_lock_id=0,
     * acquire_lock_val=0, release_lock_id=0, release_lock_val=0 */
    void *v55 = __runtime_buffer_arg(v54);
    XAie_DmaDesc v56 = __Runtime_dma_bd_config(g_DevInst, v40, v55, 1, 32, -1, 0, 0, 0, 0, 0, 0);
    /* Create IO: channel_id=1, bd_id=1, tile=(3,0), direction=MM2S */
    io v57 = __Runtime_dma_createio_4(v40, v56, 1, 1, DMA_MM2S);
    void *v58 = __runtime_buffer_offset(v54, 128);
    /* DMA BD Config: bd_id=3, offset=0, len=16, enable_packet=false, packet_id=0, next_bd=2, acquire_lock_id=2,
     * acquire_lock_val=-1, release_lock_id=3, release_lock_val=1 */
    void *v59 = __runtime_buffer_arg((void *)32960);
    XAie_DmaDesc v60 = __Runtime_dma_bd_config(g_DevInst, v25, v59, 3, 16, 2, 0, 0, 2, -1, 3, 1);
    /* Lock init: tile(0,4) lock=2 init_value=2 */
    XAie_LockSetValue(g_DevInst, XAie_TileLoc(0, 4), XAie_LockInit(2, 2));
    /* DMA BD Config: bd_id=2, offset=0, len=16, enable_packet=false, packet_id=0, next_bd=3, acquire_lock_id=2,
     * acquire_lock_val=-1, release_lock_id=3, release_lock_val=1 */
    void *v61 = __runtime_buffer_arg((void *)32896);
    XAie_DmaDesc v62 = __Runtime_dma_bd_config(g_DevInst, v25, v61, 2, 16, 3, 0, 0, 2, -1, 3, 1);
    /* Create IO: channel_id=1, bd_id=2, tile=(0,4), direction=S2MM */
    io v63 = __Runtime_dma_createio_4(v25, v62, 1, 2, DMA_S2MM);
    /* Allocated BD ID 1 for tile (0,4) */
    void *v64 = __runtime_buffer_offset(v54, 128);
    /* DMA BD Config: bd_id=3, offset=0, len=16, enable_packet=false, packet_id=0, next_bd=2, acquire_lock_id=2,
     * acquire_lock_val=-1, release_lock_id=3, release_lock_val=1 */
    void *v65 = __runtime_buffer_arg((void *)32960);
    XAie_DmaDesc v66 = __Runtime_dma_bd_config(g_DevInst, v32, v65, 3, 16, 2, 0, 0, 2, -1, 3, 1);
    /* Lock init: tile(1,4) lock=2 init_value=2 */
    XAie_LockSetValue(g_DevInst, XAie_TileLoc(1, 4), XAie_LockInit(2, 2));
    /* DMA BD Config: bd_id=2, offset=0, len=16, enable_packet=false, packet_id=0, next_bd=3, acquire_lock_id=2,
     * acquire_lock_val=-1, release_lock_id=3, release_lock_val=1 */
    void *v67 = __runtime_buffer_arg((void *)32896);
    XAie_DmaDesc v68 = __Runtime_dma_bd_config(g_DevInst, v32, v67, 2, 16, 3, 0, 0, 2, -1, 3, 1);
    /* Create IO: channel_id=1, bd_id=2, tile=(1,4), direction=S2MM */
    io v69 = __Runtime_dma_createio_4(v32, v68, 1, 2, DMA_S2MM);
    /* Allocated BD ID 1 for tile (1,4) */
    /* Allocated BD ID 1 for tile (3,0) */
    void *v70 = __runtime_buffer_offset(v3, 0);
    /* DMA BD Config: bd_id=2, offset=0, len=32, enable_packet=false, packet_id=0, next_bd=-1, acquire_lock_id=0,
     * acquire_lock_val=0, release_lock_id=0, release_lock_val=0 */
    void *v71 = __runtime_buffer_arg(v70);
    XAie_DmaDesc v72 = __Runtime_dma_bd_config(g_DevInst, v5, v71, 2, 32, -1, 0, 0, 0, 0, 0, 0);
    /* Create IO: channel_id=0, bd_id=2, tile=(2,0), direction=S2MM */
    io v73 = __Runtime_dma_createio_4(v5, v72, 0, 2, DMA_S2MM);
    void *v74 = __runtime_buffer_offset(v70, 0);
    /* DMA BD Config: bd_id=5, offset=0, len=8, enable_packet=true, packet_id=9, next_bd=4, acquire_lock_id=5,
     * acquire_lock_val=-1, release_lock_id=4, release_lock_val=1 */
    void *v75 = __runtime_buffer_arg((void *)33088);
    XAie_DmaDesc v76 = __Runtime_dma_bd_config(g_DevInst, v9, v75, 5, 8, 4, 1, 9, 5, -1, 4, 1);
    /* Lock init: tile(0,3) lock=4 init_value=2 (kernel output acquire) */
    XAie_LockSetValue(g_DevInst, XAie_TileLoc(0, 3), XAie_LockInit(4, 2));
    /* DMA BD Config: bd_id=4, offset=0, len=8, enable_packet=true, packet_id=9, next_bd=5, acquire_lock_id=5,
     * acquire_lock_val=-1, release_lock_id=4, release_lock_val=1 */
    void *v77 = __runtime_buffer_arg((void *)33024);
    XAie_DmaDesc v78 = __Runtime_dma_bd_config(g_DevInst, v9, v77, 4, 8, 5, 1, 9, 5, -1, 4, 1);
    /* Create IO: channel_id=0, bd_id=4, tile=(0,3), direction=MM2S */
    io v79 = __Runtime_dma_createio_4(v9, v78, 0, 4, DMA_MM2S);
    /* Allocated BD ID 2 for tile (0,3) */
    void *v80 = __runtime_buffer_offset(v70, 64);
    /* DMA BD Config: bd_id=5, offset=0, len=8, enable_packet=true, packet_id=10, next_bd=4, acquire_lock_id=5,
     * acquire_lock_val=-1, release_lock_id=4, release_lock_val=1 */
    void *v81 = __runtime_buffer_arg((void *)33088);
    XAie_DmaDesc v82 = __Runtime_dma_bd_config(g_DevInst, v15, v81, 5, 8, 4, 1, 10, 5, -1, 4, 1);
    /* Lock init: tile(1,3) lock=4 init_value=2 (kernel output acquire) */
    XAie_LockSetValue(g_DevInst, XAie_TileLoc(1, 3), XAie_LockInit(4, 2));
    /* DMA BD Config: bd_id=4, offset=0, len=8, enable_packet=true, packet_id=10, next_bd=5, acquire_lock_id=5,
     * acquire_lock_val=-1, release_lock_id=4, release_lock_val=1 */
    void *v83 = __runtime_buffer_arg((void *)33024);
    XAie_DmaDesc v84 = __Runtime_dma_bd_config(g_DevInst, v15, v83, 4, 8, 5, 1, 10, 5, -1, 4, 1);
    /* Create IO: channel_id=0, bd_id=4, tile=(1,3), direction=MM2S */
    io v85 = __Runtime_dma_createio_4(v15, v84, 0, 4, DMA_MM2S);
    /* Allocated BD ID 2 for tile (1,3) */
    /* Allocated BD ID 2 for tile (2,0) */
    void *v86 = __runtime_buffer_offset(v3, 128);
    /* DMA BD Config: bd_id=3, offset=0, len=32, enable_packet=false, packet_id=0, next_bd=-1, acquire_lock_id=0,
     * acquire_lock_val=0, release_lock_id=0, release_lock_val=0 */
    void *v87 = __runtime_buffer_arg(v86);
    XAie_DmaDesc v88 = __Runtime_dma_bd_config(g_DevInst, v5, v87, 3, 32, -1, 0, 0, 0, 0, 0, 0);
    /* Create IO: channel_id=1, bd_id=3, tile=(2,0), direction=S2MM */
    io v89 = __Runtime_dma_createio_4(v5, v88, 1, 3, DMA_S2MM);
    void *v90 = __runtime_buffer_offset(v86, 0);
    /* DMA BD Config: bd_id=5, offset=0, len=8, enable_packet=true, packet_id=11, next_bd=4, acquire_lock_id=5,
     * acquire_lock_val=-1, release_lock_id=4, release_lock_val=1 */
    void *v91 = __runtime_buffer_arg((void *)33088);
    XAie_DmaDesc v92 = __Runtime_dma_bd_config(g_DevInst, v25, v91, 5, 8, 4, 1, 11, 5, -1, 4, 1);
    /* Lock init: tile(0,4) lock=4 init_value=2 (kernel output acquire) */
    XAie_LockSetValue(g_DevInst, XAie_TileLoc(0, 4), XAie_LockInit(4, 2));
    /* DMA BD Config: bd_id=4, offset=0, len=8, enable_packet=true, packet_id=11, next_bd=5, acquire_lock_id=5,
     * acquire_lock_val=-1, release_lock_id=4, release_lock_val=1 */
    void *v93 = __runtime_buffer_arg((void *)33024);
    XAie_DmaDesc v94 = __Runtime_dma_bd_config(g_DevInst, v25, v93, 4, 8, 5, 1, 11, 5, -1, 4, 1);
    /* Create IO: channel_id=0, bd_id=4, tile=(0,4), direction=MM2S */
    io v95 = __Runtime_dma_createio_4(v25, v94, 0, 4, DMA_MM2S);
    /* Allocated BD ID 2 for tile (0,4) */
    void *v96 = __runtime_buffer_offset(v86, 64);
    /* DMA BD Config: bd_id=5, offset=0, len=8, enable_packet=true, packet_id=12, next_bd=4, acquire_lock_id=5,
     * acquire_lock_val=-1, release_lock_id=4, release_lock_val=1 */
    void *v97 = __runtime_buffer_arg((void *)33088);
    XAie_DmaDesc v98 = __Runtime_dma_bd_config(g_DevInst, v32, v97, 5, 8, 4, 1, 12, 5, -1, 4, 1);
    /* Lock init: tile(1,4) lock=4 init_value=2 (kernel output acquire) */
    XAie_LockSetValue(g_DevInst, XAie_TileLoc(1, 4), XAie_LockInit(4, 2));
    /* DMA BD Config: bd_id=4, offset=0, len=8, enable_packet=true, packet_id=12, next_bd=5, acquire_lock_id=5,
     * acquire_lock_val=-1, release_lock_id=4, release_lock_val=1 */
    void *v99 = __runtime_buffer_arg((void *)33024);
    XAie_DmaDesc v100 = __Runtime_dma_bd_config(g_DevInst, v32, v99, 4, 8, 5, 1, 12, 5, -1, 4, 1);
    /* Create IO: channel_id=0, bd_id=4, tile=(1,4), direction=MM2S */
    io v101 = __Runtime_dma_createio_4(v32, v100, 0, 4, DMA_MM2S);
    /* Allocated BD ID 2 for tile (1,4) */
    /* Allocated BD ID 3 for tile (2,0) */
    /* Load Kernel Group: 4 tile(s) */
    kernel_group v102 = __Runtime_load_kernel_group_4t(v9, v15, v25, v32, 4);
    /* Launch Kernel Group */
    event v103 = __Runtime_launch_kernel_group(v102);
    ioevent v104 = __Runtime_startio(v14, 0);
    ioevent v105 = __Runtime_startio(v20, 0);
    ioevent v106 = __Runtime_startio(v8, 0);
    ioevent v107 = __Runtime_startio(v31, 0);
    ioevent v108 = __Runtime_startio(v38, 0);
    ioevent v109 = __Runtime_startio(v24, 1);
    ioevent v110 = __Runtime_startio(v48, 1);
    ioevent v111 = __Runtime_startio(v53, 1);
    ioevent v112 = __Runtime_startio(v43, 0);
    ioevent v113 = __Runtime_startio(v63, 1);
    ioevent v114 = __Runtime_startio(v69, 1);
    ioevent v115 = __Runtime_startio(v57, 1);
    ioevent v116 = __Runtime_startio(v79, 2);
    ioevent v117 = __Runtime_startio(v85, 2);
    ioevent v118 = __Runtime_startio(v73, 2);
    ioevent v119 = __Runtime_startio(v95, 2);
    ioevent v120 = __Runtime_startio(v101, 2);
    ioevent v121 = __Runtime_startio(v89, 3);
    /* Wait for 7 event(s) */
    __Runtime_wait(v103);
    __Runtime_wait(v106);
    __Runtime_wait(v109);
    __Runtime_wait(v112);
    __Runtime_wait(v115);
    __Runtime_wait(v118);
    __Runtime_wait(v121);
    /* AieRt debug snapshot */
    {
        uint8_t _dbg_io_cols[] = {2, 0, 1, 2, 0, 1, 3, 0, 1, 3, 0, 1, 2, 0, 1, 2, 0, 1};
        uint8_t _dbg_io_rows[] = {0, 3, 3, 0, 4, 4, 0, 3, 3, 0, 4, 4, 0, 3, 3, 0, 4, 4};
        uint8_t _dbg_io_chs[] = {0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0};
        uint8_t _dbg_io_bds[] = {0, 0, 0, 1, 0, 0, 0, 2, 2, 1, 2, 2, 2, 4, 4, 3, 4, 4};
        int _dbg_io_dirs[] = {DMA_MM2S, DMA_S2MM, DMA_S2MM, DMA_MM2S, DMA_S2MM, DMA_S2MM, DMA_MM2S, DMA_S2MM, DMA_S2MM,
                              DMA_MM2S, DMA_S2MM, DMA_S2MM, DMA_S2MM, DMA_MM2S, DMA_MM2S, DMA_S2MM, DMA_MM2S, DMA_MM2S};
        uint8_t _dbg_t_cols[] = {0, 1, 0, 1};
        uint8_t _dbg_t_rows[] = {3, 3, 4, 4};
        AieRt_DebugSnapshotFromCoords(g_DevInst, _dbg_io_cols, _dbg_io_rows, _dbg_io_chs, _dbg_io_bds, _dbg_io_dirs, 18,
                                      _dbg_t_cols, _dbg_t_rows, 4);
    }
    return;
}

__global__ void dskernel_receiver(size_t v1) {
    // the real kernel will be emitted separately

    return;
}

// ===== User source (preserved from original file) =====
#define AIEHLC_TILING_STUBS_DEFINED
struct aieDim {
    int rows, cols;
    aieDim(int r, int c) : rows(r), cols(c) {}
};
inline void aieSetDevice(int) {}
inline void aieDeviceSynchronize() {}
inline void __aie_launch(const char *kernel, aieDim mesh, void *_t0, void *_t1, void *_t2, ...) {
    (void)kernel;
    (void)mesh;
    host_canonicalized(_t0, _t1, _t2);
}
// Stub type declarations for Clang parsing (function body skipped via #ifdef KERNEL_COMPILE)
#ifndef AIEHLC_STUBS_DEFINED
#define AIEHLC_STUBS_DEFINED
template <typename T> struct input_window {};
template <typename T> struct output_window {};
typedef int int32;
typedef input_window<int32> input_window_int32;
typedef output_window<int32> output_window_int32;
typedef signed char int8;
typedef input_window<int8> input_window_int8;
typedef output_window<int8> output_window_int8;
typedef short int16;
typedef input_window<int16> input_window_int16;
typedef output_window<int16> output_window_int16;
typedef unsigned char uint8_t;
typedef unsigned long uintptr_t;
typedef int int8_t __attribute__((mode(QI)));
typedef int int32_t __attribute__((mode(SI)));
typedef int v4int8 __attribute__((vector_size(4)));
typedef int v4int32 __attribute__((vector_size(16)));
inline unsigned get_coreid() { return 0; }
inline void klog(const char *, int) {}
template <typename T> inline void *acquire_input_window(T *) { return (void *)0; }
template <typename T> inline void *acquire_output_window(T *) { return (void *)0; }
template <typename T> inline void release_input_window(T *) {}
template <typename T> inline void release_output_window(T *) {}
#define BUF_SZ 16
#endif

// CUDA-style AIE API stubs for Clang parsing
#ifndef AIEHLC_TILING_STUBS_DEFINED
#define AIEHLC_TILING_STUBS_DEFINED
struct aieDim {
    int rows, cols;
    aieDim(int r, int c) : rows(r), cols(c) {}
};
inline void aieSetDevice(int) {}
inline void aieDeviceSynchronize() {}
extern void host_canonicalized(...);
template <typename... Args> inline void __aie_launch(const char *kernel, aieDim mesh, Args... args) {
    (void)kernel;
    (void)mesh;
    (void)sizeof...(args);
}
#endif

/******************************************************************************
 * Copyright (C) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * AIE Programming Model — Middle Ground
 *
 * CUDA concepts kept (honest mapping):
 *   __global__             - kernel runs on AIE tiles
 *   __aie_launch("kernel", mesh, )     - launch kernel across tile mesh
 *   aieDeviceSynchronize() - wait for all tiles to finish
 *   malloc/free            - plain C host memory allocation
 *
 * What the compiler handles automatically:
 *   DDR <-> tile DMA transfers, tensor partitioning, stream switch routing,
 *   buffer descriptors, lock synchronization, core load/run/wait
 *
 ******************************************************************************/
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
// #pragma aie_debug_level 2
//  ═══════════════════════════════════════════════════════════════════════════
//  KERNEL: __attribute__((annotate("__global__"))) marks this as an AIE tile kernel
//
//  C = A * B where A is [M x K], B is [K x N], C is [M x N]
//  Each tile receives its partition of the data automatically.
//  ═══════════════════════════════════════════════════════════════════════════
/*
__attribute__((annotate("__global__"))) void matmul(const int32_t *A, const int32_t *B, int32_t *C,
                       int M, int N, int K) {
#ifdef KERNEL_COMPILE

#ifdef KERNEL_COMPILE

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            int32_t sum = 0;
            for (int k = 0; k < K; k++) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }

#endif

#endif
}
*/

// Global variables for kernel: matmul
extern unsigned char _binary_kernel_matmul_start[];
extern unsigned char _binary_kernel_matmul_end[];
extern unsigned int _binary_kernel_matmul_size;

// ═══════════════════════════════════════════════════════════════════════════
// HOST
// ═══════════════════════════════════════════════════════════════════════════
int main() {
    const int M = 16, N = 16, K = 16;

    // --- Device + mesh ---
    aieSetDevice(0);
    aieDim mesh(2, 2);

    // --- Allocate host memory (plain malloc) ---
    int32_t *A = (int32_t *)malloc(M * K * sizeof(int32_t));
    int32_t *B = (int32_t *)malloc(K * N * sizeof(int32_t));
    int32_t *C = (int32_t *)malloc(M * N * sizeof(int32_t));

    for (int i = 0; i < M * K; i++)
        A[i] = i + 1;
    for (int i = 0; i < K * N; i++)
        B[i] = i + 1;
    printf("------------main--------\n");
    // --- Launch kernel on tile mesh ---
    __aie_launch("matmul", mesh, A, B, C, M, N, K);

    printf("------------after matmul--------\n");

    // --- Wait for completion ---
    aieDeviceSynchronize();

    // --- Results are ready in C ---
    int mismatches = 0;
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            int32_t expected = 0;
            for (int k = 0; k < K; k++)
                expected += A[i * K + k] * B[k * N + j];
            if (C[i * N + j] != expected) {
                printf("MISMATCH C[%d][%d]: got %d, expected %d\n", i, j, C[i * N + j], expected);
                mismatches++;
            }
        }
    }
    if (mismatches == 0)
        printf("PASS: all %d elements match.\n", M * N);
    else
        printf("FAIL: %d mismatches.\n", mismatches);

    free(A);
    free(B);
    free(C);
    return 0;
}
