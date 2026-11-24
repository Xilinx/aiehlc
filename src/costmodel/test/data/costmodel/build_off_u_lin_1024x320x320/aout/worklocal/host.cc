/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/
#include "aie_runtime.h"
#include "aie_runtime_debug.h"
#include "aie_timer.h"
XTime g_xtimer_start;
XTime g_xtimer_end;
static inline void timerStart() {
    XTime_GetTime(&g_xtimer_start);
}
static inline void timerEnd() {
    XTime_GetTime(&g_xtimer_end);
    double elapsed_ms = 1.0 * (g_xtimer_end - g_xtimer_start) / COUNTS_PER_SECOND * 1000.0;
    printf("aie kernel time: %.3f ms\n", elapsed_ms);
}
void host_canonicalized(XAie_DevInst* v1, void* v2, void* v3, void* v4) {
  size_t v5 = 1;
  size_t v6 = 16;
  size_t v7 = 0;
  int32_t v8 = 10240;
  int32_t v9 = 20480;
  int32_t v10 = 960;
  int32_t v11 = 640;
  int32_t v12 = 320;
  XAie_DevInst* dev = v1;
  void* v13 = __runtime_buffer_offset(v3, 0);
  XAie_LocType v14 = XAie_TileLoc(0, 0);
  XAie_LocType v15 = XAie_TileLoc(0, 3);
  /* Allocated BD ID 0 for tile (0,3) */
  XAie_LocType v16 = XAie_TileLoc(0, 4);
  /* Allocated BD ID 0 for tile (0,4) */
  XAie_LocType v17 = XAie_TileLoc(0, 5);
  /* Allocated BD ID 0 for tile (0,5) */
  XAie_LocType v18 = XAie_TileLoc(0, 6);
  /* Allocated BD ID 0 for tile (0,6) */
  /* Allocated BD ID 0 for tile (0,0) */
  void* v19 = __runtime_buffer_offset(v3, 51200);
  XAie_LocType v20 = XAie_TileLoc(1, 0);
  XAie_LocType v21 = XAie_TileLoc(1, 3);
  void* v22 = __runtime_buffer_offset(v19, 51200);
  /* Allocated BD ID 0 for tile (1,3) */
  XAie_LocType v23 = XAie_TileLoc(1, 4);
  void* v24 = __runtime_buffer_offset(v19, 51200);
  /* Allocated BD ID 0 for tile (1,4) */
  XAie_LocType v25 = XAie_TileLoc(1, 5);
  void* v26 = __runtime_buffer_offset(v19, 51200);
  /* Allocated BD ID 0 for tile (1,5) */
  XAie_LocType v27 = XAie_TileLoc(1, 6);
  void* v28 = __runtime_buffer_offset(v19, 51200);
  /* Allocated BD ID 0 for tile (1,6) */
  /* Allocated BD ID 0 for tile (1,0) */
  void* v29 = __runtime_buffer_offset(v3, 102400);
  XAie_LocType v30 = XAie_TileLoc(2, 0);
  XAie_LocType v31 = XAie_TileLoc(2, 3);
  void* v32 = __runtime_buffer_offset(v29, 102400);
  /* Allocated BD ID 0 for tile (2,3) */
  XAie_LocType v33 = XAie_TileLoc(2, 4);
  void* v34 = __runtime_buffer_offset(v29, 102400);
  /* Allocated BD ID 0 for tile (2,4) */
  XAie_LocType v35 = XAie_TileLoc(2, 5);
  void* v36 = __runtime_buffer_offset(v29, 102400);
  /* Allocated BD ID 0 for tile (2,5) */
  XAie_LocType v37 = XAie_TileLoc(2, 6);
  void* v38 = __runtime_buffer_offset(v29, 102400);
  /* Allocated BD ID 0 for tile (2,6) */
  /* Allocated BD ID 0 for tile (2,0) */
  void* v39 = __runtime_buffer_offset(v3, 153600);
  XAie_LocType v40 = XAie_TileLoc(3, 0);
  XAie_LocType v41 = XAie_TileLoc(3, 3);
  void* v42 = __runtime_buffer_offset(v39, 153600);
  /* Allocated BD ID 0 for tile (3,3) */
  XAie_LocType v43 = XAie_TileLoc(3, 4);
  void* v44 = __runtime_buffer_offset(v39, 153600);
  /* Allocated BD ID 0 for tile (3,4) */
  XAie_LocType v45 = XAie_TileLoc(3, 5);
  void* v46 = __runtime_buffer_offset(v39, 153600);
  /* Allocated BD ID 0 for tile (3,5) */
  XAie_LocType v47 = XAie_TileLoc(3, 6);
  void* v48 = __runtime_buffer_offset(v39, 153600);
  /* Allocated BD ID 0 for tile (3,6) */
  /* Allocated BD ID 0 for tile (3,0) */
  void* v49 = __runtime_buffer_offset(v2, 0);
  /* Allocated BD ID 1 for tile (0,3) */
  /* Allocated BD ID 1 for tile (1,3) */
  /* Allocated BD ID 1 for tile (2,3) */
  /* Allocated BD ID 1 for tile (3,3) */
  /* Allocated BD ID 1 for tile (0,0) */
  void* v50 = __runtime_buffer_offset(v4, 0);
  void* v51 = __runtime_buffer_offset(v50, 0);
  /* Allocated BD ID 2 for tile (0,3) */
  void* v52 = __runtime_buffer_offset(v50, 81920);
  /* Allocated BD ID 2 for tile (1,3) */
  void* v53 = __runtime_buffer_offset(v50, 163840);
  /* Allocated BD ID 2 for tile (2,3) */
  void* v54 = __runtime_buffer_offset(v50, 245760);
  /* Allocated BD ID 2 for tile (3,3) */
  /* Allocated BD ID 1 for tile (3,0) */
  void* v55 = __runtime_buffer_offset(v2, 163840);
  void* v56 = __runtime_buffer_offset(v55, 163840);
  /* Allocated BD ID 1 for tile (0,4) */
  void* v57 = __runtime_buffer_offset(v55, 163840);
  /* Allocated BD ID 1 for tile (1,4) */
  void* v58 = __runtime_buffer_offset(v55, 163840);
  /* Allocated BD ID 1 for tile (2,4) */
  void* v59 = __runtime_buffer_offset(v55, 163840);
  /* Allocated BD ID 1 for tile (3,4) */
  /* Allocated BD ID 1 for tile (1,0) */
  void* v60 = __runtime_buffer_offset(v4, 327680);
  void* v61 = __runtime_buffer_offset(v60, 0);
  /* Allocated BD ID 2 for tile (0,4) */
  void* v62 = __runtime_buffer_offset(v60, 81920);
  /* Allocated BD ID 2 for tile (1,4) */
  void* v63 = __runtime_buffer_offset(v60, 163840);
  /* Allocated BD ID 2 for tile (2,4) */
  void* v64 = __runtime_buffer_offset(v60, 245760);
  /* Allocated BD ID 2 for tile (3,4) */
  /* Allocated BD ID 2 for tile (3,0) */
  void* v65 = __runtime_buffer_offset(v2, 327680);
  void* v66 = __runtime_buffer_offset(v65, 327680);
  /* Allocated BD ID 1 for tile (0,5) */
  void* v67 = __runtime_buffer_offset(v65, 327680);
  /* Allocated BD ID 1 for tile (1,5) */
  void* v68 = __runtime_buffer_offset(v65, 327680);
  /* Allocated BD ID 1 for tile (2,5) */
  void* v69 = __runtime_buffer_offset(v65, 327680);
  /* Allocated BD ID 1 for tile (3,5) */
  /* Allocated BD ID 1 for tile (2,0) */
  void* v70 = __runtime_buffer_offset(v4, 655360);
  XAie_LocType v71 = XAie_TileLoc(4, 0);
  void* v72 = __runtime_buffer_offset(v70, 0);
  /* Allocated BD ID 2 for tile (0,5) */
  void* v73 = __runtime_buffer_offset(v70, 81920);
  /* Allocated BD ID 2 for tile (1,5) */
  void* v74 = __runtime_buffer_offset(v70, 163840);
  /* Allocated BD ID 2 for tile (2,5) */
  void* v75 = __runtime_buffer_offset(v70, 245760);
  /* Allocated BD ID 2 for tile (3,5) */
  /* Allocated BD ID 0 for tile (4,0) */
  void* v76 = __runtime_buffer_offset(v2, 491520);
  void* v77 = __runtime_buffer_offset(v76, 491520);
  /* Allocated BD ID 1 for tile (0,6) */
  void* v78 = __runtime_buffer_offset(v76, 491520);
  /* Allocated BD ID 1 for tile (1,6) */
  void* v79 = __runtime_buffer_offset(v76, 491520);
  /* Allocated BD ID 1 for tile (2,6) */
  void* v80 = __runtime_buffer_offset(v76, 491520);
  /* Allocated BD ID 1 for tile (3,6) */
  /* Allocated BD ID 3 for tile (3,0) */
  void* v81 = __runtime_buffer_offset(v4, 983040);
  void* v82 = __runtime_buffer_offset(v81, 0);
  /* Allocated BD ID 2 for tile (0,6) */
  void* v83 = __runtime_buffer_offset(v81, 81920);
  /* Allocated BD ID 2 for tile (1,6) */
  void* v84 = __runtime_buffer_offset(v81, 163840);
  /* Allocated BD ID 2 for tile (2,6) */
  void* v85 = __runtime_buffer_offset(v81, 245760);
  /* Allocated BD ID 2 for tile (3,6) */
  /* Allocated BD ID 1 for tile (4,0) */
  /* Load Kernel Group: 16 tile(s) */
  kernel_group v86 = __Runtime_load_kernel_group_16t(v1, v15, v16, v17, v18, v21, v23, v25, v27, v31, v33, v35, v37, v41, v43, v45, v47, 16);
  /* Lock inits — must precede kernel launch to avoid contention */
  /* Lock init: tile(0,3) lock=2 init_value=2  */
  XAie_LockSetValue(dev, XAie_TileLoc(0, 3), XAie_LockInit(2, 2));
  /* Lock init: tile(0,4) lock=2 init_value=2  */
  XAie_LockSetValue(dev, XAie_TileLoc(0, 4), XAie_LockInit(2, 2));
  /* Lock init: tile(0,5) lock=2 init_value=2  */
  XAie_LockSetValue(dev, XAie_TileLoc(0, 5), XAie_LockInit(2, 2));
  /* Lock init: tile(0,6) lock=2 init_value=2  */
  XAie_LockSetValue(dev, XAie_TileLoc(0, 6), XAie_LockInit(2, 2));
  /* Lock init: tile(1,3) lock=2 init_value=2  */
  XAie_LockSetValue(dev, XAie_TileLoc(1, 3), XAie_LockInit(2, 2));
  /* Lock init: tile(1,4) lock=2 init_value=2  */
  XAie_LockSetValue(dev, XAie_TileLoc(1, 4), XAie_LockInit(2, 2));
  /* Lock init: tile(1,5) lock=2 init_value=2  */
  XAie_LockSetValue(dev, XAie_TileLoc(1, 5), XAie_LockInit(2, 2));
  /* Lock init: tile(1,6) lock=2 init_value=2  */
  XAie_LockSetValue(dev, XAie_TileLoc(1, 6), XAie_LockInit(2, 2));
  /* Lock init: tile(2,3) lock=2 init_value=2  */
  XAie_LockSetValue(dev, XAie_TileLoc(2, 3), XAie_LockInit(2, 2));
  /* Lock init: tile(2,4) lock=2 init_value=2  */
  XAie_LockSetValue(dev, XAie_TileLoc(2, 4), XAie_LockInit(2, 2));
  /* Lock init: tile(2,5) lock=2 init_value=2  */
  XAie_LockSetValue(dev, XAie_TileLoc(2, 5), XAie_LockInit(2, 2));
  /* Lock init: tile(2,6) lock=2 init_value=2  */
  XAie_LockSetValue(dev, XAie_TileLoc(2, 6), XAie_LockInit(2, 2));
  /* Lock init: tile(3,3) lock=2 init_value=2  */
  XAie_LockSetValue(dev, XAie_TileLoc(3, 3), XAie_LockInit(2, 2));
  /* Lock init: tile(3,4) lock=2 init_value=2  */
  XAie_LockSetValue(dev, XAie_TileLoc(3, 4), XAie_LockInit(2, 2));
  /* Lock init: tile(3,5) lock=2 init_value=2  */
  XAie_LockSetValue(dev, XAie_TileLoc(3, 5), XAie_LockInit(2, 2));
  /* Lock init: tile(3,6) lock=2 init_value=2  */
  XAie_LockSetValue(dev, XAie_TileLoc(3, 6), XAie_LockInit(2, 2));
  /* Lock init: tile(0,3) lock=0 init_value=2  */
  XAie_LockSetValue(dev, XAie_TileLoc(0, 3), XAie_LockInit(0, 2));
  /* Lock init: tile(1,3) lock=0 init_value=2  */
  XAie_LockSetValue(dev, XAie_TileLoc(1, 3), XAie_LockInit(0, 2));
  /* Lock init: tile(2,3) lock=0 init_value=2  */
  XAie_LockSetValue(dev, XAie_TileLoc(2, 3), XAie_LockInit(0, 2));
  /* Lock init: tile(3,3) lock=0 init_value=2  */
  XAie_LockSetValue(dev, XAie_TileLoc(3, 3), XAie_LockInit(0, 2));
  /* Lock init: tile(0,3) lock=4 init_value=2 (kernel output acquire) */
  XAie_LockSetValue(dev, XAie_TileLoc(0, 3), XAie_LockInit(4, 2));
  /* Lock init: tile(1,3) lock=4 init_value=2 (kernel output acquire) */
  XAie_LockSetValue(dev, XAie_TileLoc(1, 3), XAie_LockInit(4, 2));
  /* Lock init: tile(2,3) lock=4 init_value=2 (kernel output acquire) */
  XAie_LockSetValue(dev, XAie_TileLoc(2, 3), XAie_LockInit(4, 2));
  /* Lock init: tile(3,3) lock=4 init_value=2 (kernel output acquire) */
  XAie_LockSetValue(dev, XAie_TileLoc(3, 3), XAie_LockInit(4, 2));
  /* Lock init: tile(0,4) lock=0 init_value=2  */
  XAie_LockSetValue(dev, XAie_TileLoc(0, 4), XAie_LockInit(0, 2));
  /* Lock init: tile(1,4) lock=0 init_value=2  */
  XAie_LockSetValue(dev, XAie_TileLoc(1, 4), XAie_LockInit(0, 2));
  /* Lock init: tile(2,4) lock=0 init_value=2  */
  XAie_LockSetValue(dev, XAie_TileLoc(2, 4), XAie_LockInit(0, 2));
  /* Lock init: tile(3,4) lock=0 init_value=2  */
  XAie_LockSetValue(dev, XAie_TileLoc(3, 4), XAie_LockInit(0, 2));
  /* Lock init: tile(0,4) lock=4 init_value=2 (kernel output acquire) */
  XAie_LockSetValue(dev, XAie_TileLoc(0, 4), XAie_LockInit(4, 2));
  /* Lock init: tile(1,4) lock=4 init_value=2 (kernel output acquire) */
  XAie_LockSetValue(dev, XAie_TileLoc(1, 4), XAie_LockInit(4, 2));
  /* Lock init: tile(2,4) lock=4 init_value=2 (kernel output acquire) */
  XAie_LockSetValue(dev, XAie_TileLoc(2, 4), XAie_LockInit(4, 2));
  /* Lock init: tile(3,4) lock=4 init_value=2 (kernel output acquire) */
  XAie_LockSetValue(dev, XAie_TileLoc(3, 4), XAie_LockInit(4, 2));
  /* Lock init: tile(0,5) lock=0 init_value=2  */
  XAie_LockSetValue(dev, XAie_TileLoc(0, 5), XAie_LockInit(0, 2));
  /* Lock init: tile(1,5) lock=0 init_value=2  */
  XAie_LockSetValue(dev, XAie_TileLoc(1, 5), XAie_LockInit(0, 2));
  /* Lock init: tile(2,5) lock=0 init_value=2  */
  XAie_LockSetValue(dev, XAie_TileLoc(2, 5), XAie_LockInit(0, 2));
  /* Lock init: tile(3,5) lock=0 init_value=2  */
  XAie_LockSetValue(dev, XAie_TileLoc(3, 5), XAie_LockInit(0, 2));
  /* Lock init: tile(0,5) lock=4 init_value=2 (kernel output acquire) */
  XAie_LockSetValue(dev, XAie_TileLoc(0, 5), XAie_LockInit(4, 2));
  /* Lock init: tile(1,5) lock=4 init_value=2 (kernel output acquire) */
  XAie_LockSetValue(dev, XAie_TileLoc(1, 5), XAie_LockInit(4, 2));
  /* Lock init: tile(2,5) lock=4 init_value=2 (kernel output acquire) */
  XAie_LockSetValue(dev, XAie_TileLoc(2, 5), XAie_LockInit(4, 2));
  /* Lock init: tile(3,5) lock=4 init_value=2 (kernel output acquire) */
  XAie_LockSetValue(dev, XAie_TileLoc(3, 5), XAie_LockInit(4, 2));
  /* Lock init: tile(0,6) lock=0 init_value=2  */
  XAie_LockSetValue(dev, XAie_TileLoc(0, 6), XAie_LockInit(0, 2));
  /* Lock init: tile(1,6) lock=0 init_value=2  */
  XAie_LockSetValue(dev, XAie_TileLoc(1, 6), XAie_LockInit(0, 2));
  /* Lock init: tile(2,6) lock=0 init_value=2  */
  XAie_LockSetValue(dev, XAie_TileLoc(2, 6), XAie_LockInit(0, 2));
  /* Lock init: tile(3,6) lock=0 init_value=2  */
  XAie_LockSetValue(dev, XAie_TileLoc(3, 6), XAie_LockInit(0, 2));
  /* Lock init: tile(0,6) lock=4 init_value=2 (kernel output acquire) */
  XAie_LockSetValue(dev, XAie_TileLoc(0, 6), XAie_LockInit(4, 2));
  /* Lock init: tile(1,6) lock=4 init_value=2 (kernel output acquire) */
  XAie_LockSetValue(dev, XAie_TileLoc(1, 6), XAie_LockInit(4, 2));
  /* Lock init: tile(2,6) lock=4 init_value=2 (kernel output acquire) */
  XAie_LockSetValue(dev, XAie_TileLoc(2, 6), XAie_LockInit(4, 2));
  /* Lock init: tile(3,6) lock=4 init_value=2 (kernel output acquire) */
  XAie_LockSetValue(dev, XAie_TileLoc(3, 6), XAie_LockInit(4, 2));
  /* Launch Kernel Group */
  event v87 = __Runtime_launch_kernel_group(v1, v86);
  /* DMA BD Config: bd_id=1, len=2560, enable_packet=false, packet_id=0, next_bd=0, acquire_lock_id=2, acquire_lock_val=-1, release_lock_id=3, release_lock_val=1, ooo_bd_id=-1 */
  void* v88 = __runtime_buffer_arg((void*)35328);
  XAie_DmaDesc v89 = __Runtime_dma_bd_config(v1, v15, v88, 1, 2560, 0, 0, 0, 2, -1, 3, 1, -1);
  /* DMA BD Config: bd_id=0, len=2560, enable_packet=false, packet_id=0, next_bd=1, acquire_lock_id=2, acquire_lock_val=-1, release_lock_id=3, release_lock_val=1, ooo_bd_id=-1 */
  void* v90 = __runtime_buffer_arg((void*)32768);
  XAie_DmaDesc v91 = __Runtime_dma_bd_config(v1, v15, v90, 0, 2560, 1, 0, 0, 2, -1, 3, 1, -1);
  /* Create IO: channel_id=1, bd_id=0, tile=(0,3), direction=S2MM */
  io v92 = __Runtime_dma_createio_4(v15, v91, 1, 0, DMA_S2MM);
  ioevent v93 = __Runtime_startio(v1, v92, 0, 1);
  /* DMA BD Config: bd_id=1, len=2560, enable_packet=false, packet_id=0, next_bd=0, acquire_lock_id=2, acquire_lock_val=-1, release_lock_id=3, release_lock_val=1, ooo_bd_id=-1 */
  void* v94 = __runtime_buffer_arg((void*)35328);
  XAie_DmaDesc v95 = __Runtime_dma_bd_config(v1, v16, v94, 1, 2560, 0, 0, 0, 2, -1, 3, 1, -1);
  /* DMA BD Config: bd_id=0, len=2560, enable_packet=false, packet_id=0, next_bd=1, acquire_lock_id=2, acquire_lock_val=-1, release_lock_id=3, release_lock_val=1, ooo_bd_id=-1 */
  void* v96 = __runtime_buffer_arg((void*)32768);
  XAie_DmaDesc v97 = __Runtime_dma_bd_config(v1, v16, v96, 0, 2560, 1, 0, 0, 2, -1, 3, 1, -1);
  /* Create IO: channel_id=1, bd_id=0, tile=(0,4), direction=S2MM */
  io v98 = __Runtime_dma_createio_4(v16, v97, 1, 0, DMA_S2MM);
  ioevent v99 = __Runtime_startio(v1, v98, 0, 1);
  /* DMA BD Config: bd_id=1, len=2560, enable_packet=false, packet_id=0, next_bd=0, acquire_lock_id=2, acquire_lock_val=-1, release_lock_id=3, release_lock_val=1, ooo_bd_id=-1 */
  void* v100 = __runtime_buffer_arg((void*)35328);
  XAie_DmaDesc v101 = __Runtime_dma_bd_config(v1, v17, v100, 1, 2560, 0, 0, 0, 2, -1, 3, 1, -1);
  /* DMA BD Config: bd_id=0, len=2560, enable_packet=false, packet_id=0, next_bd=1, acquire_lock_id=2, acquire_lock_val=-1, release_lock_id=3, release_lock_val=1, ooo_bd_id=-1 */
  void* v102 = __runtime_buffer_arg((void*)32768);
  XAie_DmaDesc v103 = __Runtime_dma_bd_config(v1, v17, v102, 0, 2560, 1, 0, 0, 2, -1, 3, 1, -1);
  /* Create IO: channel_id=1, bd_id=0, tile=(0,5), direction=S2MM */
  io v104 = __Runtime_dma_createio_4(v17, v103, 1, 0, DMA_S2MM);
  ioevent v105 = __Runtime_startio(v1, v104, 0, 1);
  /* DMA BD Config: bd_id=1, len=2560, enable_packet=false, packet_id=0, next_bd=0, acquire_lock_id=2, acquire_lock_val=-1, release_lock_id=3, release_lock_val=1, ooo_bd_id=-1 */
  void* v106 = __runtime_buffer_arg((void*)35328);
  XAie_DmaDesc v107 = __Runtime_dma_bd_config(v1, v18, v106, 1, 2560, 0, 0, 0, 2, -1, 3, 1, -1);
  /* DMA BD Config: bd_id=0, len=2560, enable_packet=false, packet_id=0, next_bd=1, acquire_lock_id=2, acquire_lock_val=-1, release_lock_id=3, release_lock_val=1, ooo_bd_id=-1 */
  void* v108 = __runtime_buffer_arg((void*)32768);
  XAie_DmaDesc v109 = __Runtime_dma_bd_config(v1, v18, v108, 0, 2560, 1, 0, 0, 2, -1, 3, 1, -1);
  /* Create IO: channel_id=1, bd_id=0, tile=(0,6), direction=S2MM */
  io v110 = __Runtime_dma_createio_4(v18, v109, 1, 0, DMA_S2MM);
  ioevent v111 = __Runtime_startio(v1, v110, 0, 1);
  /* DMA BD Config: bd_id=1, len=2560, enable_packet=false, packet_id=0, next_bd=0, acquire_lock_id=2, acquire_lock_val=-1, release_lock_id=3, release_lock_val=1, ooo_bd_id=-1 */
  void* v112 = __runtime_buffer_arg((void*)35328);
  XAie_DmaDesc v113 = __Runtime_dma_bd_config(v1, v21, v112, 1, 2560, 0, 0, 0, 2, -1, 3, 1, -1);
  /* DMA BD Config: bd_id=0, len=2560, enable_packet=false, packet_id=0, next_bd=1, acquire_lock_id=2, acquire_lock_val=-1, release_lock_id=3, release_lock_val=1, ooo_bd_id=-1 */
  void* v114 = __runtime_buffer_arg((void*)32768);
  XAie_DmaDesc v115 = __Runtime_dma_bd_config(v1, v21, v114, 0, 2560, 1, 0, 0, 2, -1, 3, 1, -1);
  /* Create IO: channel_id=1, bd_id=0, tile=(1,3), direction=S2MM */
  io v116 = __Runtime_dma_createio_4(v21, v115, 1, 0, DMA_S2MM);
  ioevent v117 = __Runtime_startio(v1, v116, 0, 1);
  /* DMA BD Config: bd_id=1, len=2560, enable_packet=false, packet_id=0, next_bd=0, acquire_lock_id=2, acquire_lock_val=-1, release_lock_id=3, release_lock_val=1, ooo_bd_id=-1 */
  void* v118 = __runtime_buffer_arg((void*)35328);
  XAie_DmaDesc v119 = __Runtime_dma_bd_config(v1, v23, v118, 1, 2560, 0, 0, 0, 2, -1, 3, 1, -1);
  /* DMA BD Config: bd_id=0, len=2560, enable_packet=false, packet_id=0, next_bd=1, acquire_lock_id=2, acquire_lock_val=-1, release_lock_id=3, release_lock_val=1, ooo_bd_id=-1 */
  void* v120 = __runtime_buffer_arg((void*)32768);
  XAie_DmaDesc v121 = __Runtime_dma_bd_config(v1, v23, v120, 0, 2560, 1, 0, 0, 2, -1, 3, 1, -1);
  /* Create IO: channel_id=1, bd_id=0, tile=(1,4), direction=S2MM */
  io v122 = __Runtime_dma_createio_4(v23, v121, 1, 0, DMA_S2MM);
  ioevent v123 = __Runtime_startio(v1, v122, 0, 1);
  /* DMA BD Config: bd_id=1, len=2560, enable_packet=false, packet_id=0, next_bd=0, acquire_lock_id=2, acquire_lock_val=-1, release_lock_id=3, release_lock_val=1, ooo_bd_id=-1 */
  void* v124 = __runtime_buffer_arg((void*)35328);
  XAie_DmaDesc v125 = __Runtime_dma_bd_config(v1, v25, v124, 1, 2560, 0, 0, 0, 2, -1, 3, 1, -1);
  /* DMA BD Config: bd_id=0, len=2560, enable_packet=false, packet_id=0, next_bd=1, acquire_lock_id=2, acquire_lock_val=-1, release_lock_id=3, release_lock_val=1, ooo_bd_id=-1 */
  void* v126 = __runtime_buffer_arg((void*)32768);
  XAie_DmaDesc v127 = __Runtime_dma_bd_config(v1, v25, v126, 0, 2560, 1, 0, 0, 2, -1, 3, 1, -1);
  /* Create IO: channel_id=1, bd_id=0, tile=(1,5), direction=S2MM */
  io v128 = __Runtime_dma_createio_4(v25, v127, 1, 0, DMA_S2MM);
  ioevent v129 = __Runtime_startio(v1, v128, 0, 1);
  /* DMA BD Config: bd_id=1, len=2560, enable_packet=false, packet_id=0, next_bd=0, acquire_lock_id=2, acquire_lock_val=-1, release_lock_id=3, release_lock_val=1, ooo_bd_id=-1 */
  void* v130 = __runtime_buffer_arg((void*)35328);
  XAie_DmaDesc v131 = __Runtime_dma_bd_config(v1, v27, v130, 1, 2560, 0, 0, 0, 2, -1, 3, 1, -1);
  /* DMA BD Config: bd_id=0, len=2560, enable_packet=false, packet_id=0, next_bd=1, acquire_lock_id=2, acquire_lock_val=-1, release_lock_id=3, release_lock_val=1, ooo_bd_id=-1 */
  void* v132 = __runtime_buffer_arg((void*)32768);
  XAie_DmaDesc v133 = __Runtime_dma_bd_config(v1, v27, v132, 0, 2560, 1, 0, 0, 2, -1, 3, 1, -1);
  /* Create IO: channel_id=1, bd_id=0, tile=(1,6), direction=S2MM */
  io v134 = __Runtime_dma_createio_4(v27, v133, 1, 0, DMA_S2MM);
  ioevent v135 = __Runtime_startio(v1, v134, 0, 1);
  /* DMA BD Config: bd_id=1, len=2560, enable_packet=false, packet_id=0, next_bd=0, acquire_lock_id=2, acquire_lock_val=-1, release_lock_id=3, release_lock_val=1, ooo_bd_id=-1 */
  void* v136 = __runtime_buffer_arg((void*)35328);
  XAie_DmaDesc v137 = __Runtime_dma_bd_config(v1, v31, v136, 1, 2560, 0, 0, 0, 2, -1, 3, 1, -1);
  /* DMA BD Config: bd_id=0, len=2560, enable_packet=false, packet_id=0, next_bd=1, acquire_lock_id=2, acquire_lock_val=-1, release_lock_id=3, release_lock_val=1, ooo_bd_id=-1 */
  void* v138 = __runtime_buffer_arg((void*)32768);
  XAie_DmaDesc v139 = __Runtime_dma_bd_config(v1, v31, v138, 0, 2560, 1, 0, 0, 2, -1, 3, 1, -1);
  /* Create IO: channel_id=1, bd_id=0, tile=(2,3), direction=S2MM */
  io v140 = __Runtime_dma_createio_4(v31, v139, 1, 0, DMA_S2MM);
  ioevent v141 = __Runtime_startio(v1, v140, 0, 1);
  /* DMA BD Config: bd_id=1, len=2560, enable_packet=false, packet_id=0, next_bd=0, acquire_lock_id=2, acquire_lock_val=-1, release_lock_id=3, release_lock_val=1, ooo_bd_id=-1 */
  void* v142 = __runtime_buffer_arg((void*)35328);
  XAie_DmaDesc v143 = __Runtime_dma_bd_config(v1, v33, v142, 1, 2560, 0, 0, 0, 2, -1, 3, 1, -1);
  /* DMA BD Config: bd_id=0, len=2560, enable_packet=false, packet_id=0, next_bd=1, acquire_lock_id=2, acquire_lock_val=-1, release_lock_id=3, release_lock_val=1, ooo_bd_id=-1 */
  void* v144 = __runtime_buffer_arg((void*)32768);
  XAie_DmaDesc v145 = __Runtime_dma_bd_config(v1, v33, v144, 0, 2560, 1, 0, 0, 2, -1, 3, 1, -1);
  /* Create IO: channel_id=1, bd_id=0, tile=(2,4), direction=S2MM */
  io v146 = __Runtime_dma_createio_4(v33, v145, 1, 0, DMA_S2MM);
  ioevent v147 = __Runtime_startio(v1, v146, 0, 1);
  /* DMA BD Config: bd_id=1, len=2560, enable_packet=false, packet_id=0, next_bd=0, acquire_lock_id=2, acquire_lock_val=-1, release_lock_id=3, release_lock_val=1, ooo_bd_id=-1 */
  void* v148 = __runtime_buffer_arg((void*)35328);
  XAie_DmaDesc v149 = __Runtime_dma_bd_config(v1, v35, v148, 1, 2560, 0, 0, 0, 2, -1, 3, 1, -1);
  /* DMA BD Config: bd_id=0, len=2560, enable_packet=false, packet_id=0, next_bd=1, acquire_lock_id=2, acquire_lock_val=-1, release_lock_id=3, release_lock_val=1, ooo_bd_id=-1 */
  void* v150 = __runtime_buffer_arg((void*)32768);
  XAie_DmaDesc v151 = __Runtime_dma_bd_config(v1, v35, v150, 0, 2560, 1, 0, 0, 2, -1, 3, 1, -1);
  /* Create IO: channel_id=1, bd_id=0, tile=(2,5), direction=S2MM */
  io v152 = __Runtime_dma_createio_4(v35, v151, 1, 0, DMA_S2MM);
  ioevent v153 = __Runtime_startio(v1, v152, 0, 1);
  /* DMA BD Config: bd_id=1, len=2560, enable_packet=false, packet_id=0, next_bd=0, acquire_lock_id=2, acquire_lock_val=-1, release_lock_id=3, release_lock_val=1, ooo_bd_id=-1 */
  void* v154 = __runtime_buffer_arg((void*)35328);
  XAie_DmaDesc v155 = __Runtime_dma_bd_config(v1, v37, v154, 1, 2560, 0, 0, 0, 2, -1, 3, 1, -1);
  /* DMA BD Config: bd_id=0, len=2560, enable_packet=false, packet_id=0, next_bd=1, acquire_lock_id=2, acquire_lock_val=-1, release_lock_id=3, release_lock_val=1, ooo_bd_id=-1 */
  void* v156 = __runtime_buffer_arg((void*)32768);
  XAie_DmaDesc v157 = __Runtime_dma_bd_config(v1, v37, v156, 0, 2560, 1, 0, 0, 2, -1, 3, 1, -1);
  /* Create IO: channel_id=1, bd_id=0, tile=(2,6), direction=S2MM */
  io v158 = __Runtime_dma_createio_4(v37, v157, 1, 0, DMA_S2MM);
  ioevent v159 = __Runtime_startio(v1, v158, 0, 1);
  /* DMA BD Config: bd_id=1, len=2560, enable_packet=false, packet_id=0, next_bd=0, acquire_lock_id=2, acquire_lock_val=-1, release_lock_id=3, release_lock_val=1, ooo_bd_id=-1 */
  void* v160 = __runtime_buffer_arg((void*)35328);
  XAie_DmaDesc v161 = __Runtime_dma_bd_config(v1, v41, v160, 1, 2560, 0, 0, 0, 2, -1, 3, 1, -1);
  /* DMA BD Config: bd_id=0, len=2560, enable_packet=false, packet_id=0, next_bd=1, acquire_lock_id=2, acquire_lock_val=-1, release_lock_id=3, release_lock_val=1, ooo_bd_id=-1 */
  void* v162 = __runtime_buffer_arg((void*)32768);
  XAie_DmaDesc v163 = __Runtime_dma_bd_config(v1, v41, v162, 0, 2560, 1, 0, 0, 2, -1, 3, 1, -1);
  /* Create IO: channel_id=1, bd_id=0, tile=(3,3), direction=S2MM */
  io v164 = __Runtime_dma_createio_4(v41, v163, 1, 0, DMA_S2MM);
  ioevent v165 = __Runtime_startio(v1, v164, 0, 1);
  /* DMA BD Config: bd_id=1, len=2560, enable_packet=false, packet_id=0, next_bd=0, acquire_lock_id=2, acquire_lock_val=-1, release_lock_id=3, release_lock_val=1, ooo_bd_id=-1 */
  void* v166 = __runtime_buffer_arg((void*)35328);
  XAie_DmaDesc v167 = __Runtime_dma_bd_config(v1, v43, v166, 1, 2560, 0, 0, 0, 2, -1, 3, 1, -1);
  /* DMA BD Config: bd_id=0, len=2560, enable_packet=false, packet_id=0, next_bd=1, acquire_lock_id=2, acquire_lock_val=-1, release_lock_id=3, release_lock_val=1, ooo_bd_id=-1 */
  void* v168 = __runtime_buffer_arg((void*)32768);
  XAie_DmaDesc v169 = __Runtime_dma_bd_config(v1, v43, v168, 0, 2560, 1, 0, 0, 2, -1, 3, 1, -1);
  /* Create IO: channel_id=1, bd_id=0, tile=(3,4), direction=S2MM */
  io v170 = __Runtime_dma_createio_4(v43, v169, 1, 0, DMA_S2MM);
  ioevent v171 = __Runtime_startio(v1, v170, 0, 1);
  /* DMA BD Config: bd_id=1, len=2560, enable_packet=false, packet_id=0, next_bd=0, acquire_lock_id=2, acquire_lock_val=-1, release_lock_id=3, release_lock_val=1, ooo_bd_id=-1 */
  void* v172 = __runtime_buffer_arg((void*)35328);
  XAie_DmaDesc v173 = __Runtime_dma_bd_config(v1, v45, v172, 1, 2560, 0, 0, 0, 2, -1, 3, 1, -1);
  /* DMA BD Config: bd_id=0, len=2560, enable_packet=false, packet_id=0, next_bd=1, acquire_lock_id=2, acquire_lock_val=-1, release_lock_id=3, release_lock_val=1, ooo_bd_id=-1 */
  void* v174 = __runtime_buffer_arg((void*)32768);
  XAie_DmaDesc v175 = __Runtime_dma_bd_config(v1, v45, v174, 0, 2560, 1, 0, 0, 2, -1, 3, 1, -1);
  /* Create IO: channel_id=1, bd_id=0, tile=(3,5), direction=S2MM */
  io v176 = __Runtime_dma_createio_4(v45, v175, 1, 0, DMA_S2MM);
  ioevent v177 = __Runtime_startio(v1, v176, 0, 1);
  /* DMA BD Config: bd_id=1, len=2560, enable_packet=false, packet_id=0, next_bd=0, acquire_lock_id=2, acquire_lock_val=-1, release_lock_id=3, release_lock_val=1, ooo_bd_id=-1 */
  void* v178 = __runtime_buffer_arg((void*)35328);
  XAie_DmaDesc v179 = __Runtime_dma_bd_config(v1, v47, v178, 1, 2560, 0, 0, 0, 2, -1, 3, 1, -1);
  /* DMA BD Config: bd_id=0, len=2560, enable_packet=false, packet_id=0, next_bd=1, acquire_lock_id=2, acquire_lock_val=-1, release_lock_id=3, release_lock_val=1, ooo_bd_id=-1 */
  void* v180 = __runtime_buffer_arg((void*)32768);
  XAie_DmaDesc v181 = __Runtime_dma_bd_config(v1, v47, v180, 0, 2560, 1, 0, 0, 2, -1, 3, 1, -1);
  /* Create IO: channel_id=1, bd_id=0, tile=(3,6), direction=S2MM */
  io v182 = __Runtime_dma_createio_4(v47, v181, 1, 0, DMA_S2MM);
  ioevent v183 = __Runtime_startio(v1, v182, 0, 1);
  /* DMA BD Config: bd_id=3, len=512, enable_packet=false, packet_id=0, next_bd=2, acquire_lock_id=0, acquire_lock_val=-1, release_lock_id=1, release_lock_val=1, ooo_bd_id=-1 */
  void* v184 = __runtime_buffer_arg((void*)38400);
  XAie_DmaDesc v185 = __Runtime_dma_bd_config(v1, v15, v184, 3, 512, 2, 0, 0, 0, -1, 1, 1, -1);
  /* DMA BD Config: bd_id=2, len=512, enable_packet=false, packet_id=0, next_bd=3, acquire_lock_id=0, acquire_lock_val=-1, release_lock_id=1, release_lock_val=1, ooo_bd_id=-1 */
  void* v186 = __runtime_buffer_arg((void*)37888);
  XAie_DmaDesc v187 = __Runtime_dma_bd_config(v1, v15, v186, 2, 512, 3, 0, 0, 0, -1, 1, 1, -1);
  /* Create IO: channel_id=0, bd_id=2, tile=(0,3), direction=S2MM */
  io v188 = __Runtime_dma_createio_4(v15, v187, 0, 2, DMA_S2MM);
  ioevent v189 = __Runtime_startio(v1, v188, 1, 1);
  /* DMA BD Config: bd_id=3, len=512, enable_packet=false, packet_id=0, next_bd=2, acquire_lock_id=0, acquire_lock_val=-1, release_lock_id=1, release_lock_val=1, ooo_bd_id=-1 */
  void* v190 = __runtime_buffer_arg((void*)38400);
  XAie_DmaDesc v191 = __Runtime_dma_bd_config(v1, v21, v190, 3, 512, 2, 0, 0, 0, -1, 1, 1, -1);
  /* DMA BD Config: bd_id=2, len=512, enable_packet=false, packet_id=0, next_bd=3, acquire_lock_id=0, acquire_lock_val=-1, release_lock_id=1, release_lock_val=1, ooo_bd_id=-1 */
  void* v192 = __runtime_buffer_arg((void*)37888);
  XAie_DmaDesc v193 = __Runtime_dma_bd_config(v1, v21, v192, 2, 512, 3, 0, 0, 0, -1, 1, 1, -1);
  /* Create IO: channel_id=0, bd_id=2, tile=(1,3), direction=S2MM */
  io v194 = __Runtime_dma_createio_4(v21, v193, 0, 2, DMA_S2MM);
  ioevent v195 = __Runtime_startio(v1, v194, 1, 1);
  /* DMA BD Config: bd_id=3, len=512, enable_packet=false, packet_id=0, next_bd=2, acquire_lock_id=0, acquire_lock_val=-1, release_lock_id=1, release_lock_val=1, ooo_bd_id=-1 */
  void* v196 = __runtime_buffer_arg((void*)38400);
  XAie_DmaDesc v197 = __Runtime_dma_bd_config(v1, v31, v196, 3, 512, 2, 0, 0, 0, -1, 1, 1, -1);
  /* DMA BD Config: bd_id=2, len=512, enable_packet=false, packet_id=0, next_bd=3, acquire_lock_id=0, acquire_lock_val=-1, release_lock_id=1, release_lock_val=1, ooo_bd_id=-1 */
  void* v198 = __runtime_buffer_arg((void*)37888);
  XAie_DmaDesc v199 = __Runtime_dma_bd_config(v1, v31, v198, 2, 512, 3, 0, 0, 0, -1, 1, 1, -1);
  /* Create IO: channel_id=0, bd_id=2, tile=(2,3), direction=S2MM */
  io v200 = __Runtime_dma_createio_4(v31, v199, 0, 2, DMA_S2MM);
  ioevent v201 = __Runtime_startio(v1, v200, 1, 1);
  /* DMA BD Config: bd_id=3, len=512, enable_packet=false, packet_id=0, next_bd=2, acquire_lock_id=0, acquire_lock_val=-1, release_lock_id=1, release_lock_val=1, ooo_bd_id=-1 */
  void* v202 = __runtime_buffer_arg((void*)38400);
  XAie_DmaDesc v203 = __Runtime_dma_bd_config(v1, v41, v202, 3, 512, 2, 0, 0, 0, -1, 1, 1, -1);
  /* DMA BD Config: bd_id=2, len=512, enable_packet=false, packet_id=0, next_bd=3, acquire_lock_id=0, acquire_lock_val=-1, release_lock_id=1, release_lock_val=1, ooo_bd_id=-1 */
  void* v204 = __runtime_buffer_arg((void*)37888);
  XAie_DmaDesc v205 = __Runtime_dma_bd_config(v1, v41, v204, 2, 512, 3, 0, 0, 0, -1, 1, 1, -1);
  /* Create IO: channel_id=0, bd_id=2, tile=(3,3), direction=S2MM */
  io v206 = __Runtime_dma_createio_4(v41, v205, 0, 2, DMA_S2MM);
  ioevent v207 = __Runtime_startio(v1, v206, 1, 1);
  /* DMA BD Config: bd_id=5, len=5120, enable_packet=true, packet_id=1, next_bd=4, acquire_lock_id=5, acquire_lock_val=-1, release_lock_id=4, release_lock_val=1, ooo_bd_id=2 */
  void* v208 = __runtime_buffer_arg((void*)44032);
  XAie_DmaDesc v209 = __Runtime_dma_bd_config(v1, v15, v208, 5, 5120, 4, 1, 1, 5, -1, 4, 1, 2);
  /* DMA BD Config: bd_id=4, len=5120, enable_packet=true, packet_id=1, next_bd=5, acquire_lock_id=5, acquire_lock_val=-1, release_lock_id=4, release_lock_val=1, ooo_bd_id=2 */
  void* v210 = __runtime_buffer_arg((void*)38912);
  XAie_DmaDesc v211 = __Runtime_dma_bd_config(v1, v15, v210, 4, 5120, 5, 1, 1, 5, -1, 4, 1, 2);
  /* Create IO: channel_id=0, bd_id=4, tile=(0,3), direction=MM2S */
  io v212 = __Runtime_dma_createio_4(v15, v211, 0, 4, DMA_MM2S);
  ioevent v213 = __Runtime_startio(v1, v212, 2, 1);
  /* DMA BD Config: bd_id=5, len=5120, enable_packet=true, packet_id=2, next_bd=4, acquire_lock_id=5, acquire_lock_val=-1, release_lock_id=4, release_lock_val=1, ooo_bd_id=3 */
  void* v214 = __runtime_buffer_arg((void*)44032);
  XAie_DmaDesc v215 = __Runtime_dma_bd_config(v1, v21, v214, 5, 5120, 4, 1, 2, 5, -1, 4, 1, 3);
  /* DMA BD Config: bd_id=4, len=5120, enable_packet=true, packet_id=2, next_bd=5, acquire_lock_id=5, acquire_lock_val=-1, release_lock_id=4, release_lock_val=1, ooo_bd_id=3 */
  void* v216 = __runtime_buffer_arg((void*)38912);
  XAie_DmaDesc v217 = __Runtime_dma_bd_config(v1, v21, v216, 4, 5120, 5, 1, 2, 5, -1, 4, 1, 3);
  /* Create IO: channel_id=0, bd_id=4, tile=(1,3), direction=MM2S */
  io v218 = __Runtime_dma_createio_4(v21, v217, 0, 4, DMA_MM2S);
  ioevent v219 = __Runtime_startio(v1, v218, 2, 1);
  /* DMA BD Config: bd_id=5, len=5120, enable_packet=true, packet_id=3, next_bd=4, acquire_lock_id=5, acquire_lock_val=-1, release_lock_id=4, release_lock_val=1, ooo_bd_id=4 */
  void* v220 = __runtime_buffer_arg((void*)44032);
  XAie_DmaDesc v221 = __Runtime_dma_bd_config(v1, v31, v220, 5, 5120, 4, 1, 3, 5, -1, 4, 1, 4);
  /* DMA BD Config: bd_id=4, len=5120, enable_packet=true, packet_id=3, next_bd=5, acquire_lock_id=5, acquire_lock_val=-1, release_lock_id=4, release_lock_val=1, ooo_bd_id=4 */
  void* v222 = __runtime_buffer_arg((void*)38912);
  XAie_DmaDesc v223 = __Runtime_dma_bd_config(v1, v31, v222, 4, 5120, 5, 1, 3, 5, -1, 4, 1, 4);
  /* Create IO: channel_id=0, bd_id=4, tile=(2,3), direction=MM2S */
  io v224 = __Runtime_dma_createio_4(v31, v223, 0, 4, DMA_MM2S);
  ioevent v225 = __Runtime_startio(v1, v224, 2, 1);
  /* DMA BD Config: bd_id=5, len=5120, enable_packet=true, packet_id=4, next_bd=4, acquire_lock_id=5, acquire_lock_val=-1, release_lock_id=4, release_lock_val=1, ooo_bd_id=5 */
  void* v226 = __runtime_buffer_arg((void*)44032);
  XAie_DmaDesc v227 = __Runtime_dma_bd_config(v1, v41, v226, 5, 5120, 4, 1, 4, 5, -1, 4, 1, 5);
  /* DMA BD Config: bd_id=4, len=5120, enable_packet=true, packet_id=4, next_bd=5, acquire_lock_id=5, acquire_lock_val=-1, release_lock_id=4, release_lock_val=1, ooo_bd_id=5 */
  void* v228 = __runtime_buffer_arg((void*)38912);
  XAie_DmaDesc v229 = __Runtime_dma_bd_config(v1, v41, v228, 4, 5120, 5, 1, 4, 5, -1, 4, 1, 5);
  /* Create IO: channel_id=0, bd_id=4, tile=(3,3), direction=MM2S */
  io v230 = __Runtime_dma_createio_4(v41, v229, 0, 4, DMA_MM2S);
  ioevent v231 = __Runtime_startio(v1, v230, 2, 1);
  /* DMA BD Config: bd_id=3, len=512, enable_packet=false, packet_id=0, next_bd=2, acquire_lock_id=0, acquire_lock_val=-1, release_lock_id=1, release_lock_val=1, ooo_bd_id=-1 */
  void* v232 = __runtime_buffer_arg((void*)38400);
  XAie_DmaDesc v233 = __Runtime_dma_bd_config(v1, v16, v232, 3, 512, 2, 0, 0, 0, -1, 1, 1, -1);
  /* DMA BD Config: bd_id=2, len=512, enable_packet=false, packet_id=0, next_bd=3, acquire_lock_id=0, acquire_lock_val=-1, release_lock_id=1, release_lock_val=1, ooo_bd_id=-1 */
  void* v234 = __runtime_buffer_arg((void*)37888);
  XAie_DmaDesc v235 = __Runtime_dma_bd_config(v1, v16, v234, 2, 512, 3, 0, 0, 0, -1, 1, 1, -1);
  /* Create IO: channel_id=0, bd_id=2, tile=(0,4), direction=S2MM */
  io v236 = __Runtime_dma_createio_4(v16, v235, 0, 2, DMA_S2MM);
  ioevent v237 = __Runtime_startio(v1, v236, 1, 1);
  /* DMA BD Config: bd_id=3, len=512, enable_packet=false, packet_id=0, next_bd=2, acquire_lock_id=0, acquire_lock_val=-1, release_lock_id=1, release_lock_val=1, ooo_bd_id=-1 */
  void* v238 = __runtime_buffer_arg((void*)38400);
  XAie_DmaDesc v239 = __Runtime_dma_bd_config(v1, v23, v238, 3, 512, 2, 0, 0, 0, -1, 1, 1, -1);
  /* DMA BD Config: bd_id=2, len=512, enable_packet=false, packet_id=0, next_bd=3, acquire_lock_id=0, acquire_lock_val=-1, release_lock_id=1, release_lock_val=1, ooo_bd_id=-1 */
  void* v240 = __runtime_buffer_arg((void*)37888);
  XAie_DmaDesc v241 = __Runtime_dma_bd_config(v1, v23, v240, 2, 512, 3, 0, 0, 0, -1, 1, 1, -1);
  /* Create IO: channel_id=0, bd_id=2, tile=(1,4), direction=S2MM */
  io v242 = __Runtime_dma_createio_4(v23, v241, 0, 2, DMA_S2MM);
  ioevent v243 = __Runtime_startio(v1, v242, 1, 1);
  /* DMA BD Config: bd_id=3, len=512, enable_packet=false, packet_id=0, next_bd=2, acquire_lock_id=0, acquire_lock_val=-1, release_lock_id=1, release_lock_val=1, ooo_bd_id=-1 */
  void* v244 = __runtime_buffer_arg((void*)38400);
  XAie_DmaDesc v245 = __Runtime_dma_bd_config(v1, v33, v244, 3, 512, 2, 0, 0, 0, -1, 1, 1, -1);
  /* DMA BD Config: bd_id=2, len=512, enable_packet=false, packet_id=0, next_bd=3, acquire_lock_id=0, acquire_lock_val=-1, release_lock_id=1, release_lock_val=1, ooo_bd_id=-1 */
  void* v246 = __runtime_buffer_arg((void*)37888);
  XAie_DmaDesc v247 = __Runtime_dma_bd_config(v1, v33, v246, 2, 512, 3, 0, 0, 0, -1, 1, 1, -1);
  /* Create IO: channel_id=0, bd_id=2, tile=(2,4), direction=S2MM */
  io v248 = __Runtime_dma_createio_4(v33, v247, 0, 2, DMA_S2MM);
  ioevent v249 = __Runtime_startio(v1, v248, 1, 1);
  /* DMA BD Config: bd_id=3, len=512, enable_packet=false, packet_id=0, next_bd=2, acquire_lock_id=0, acquire_lock_val=-1, release_lock_id=1, release_lock_val=1, ooo_bd_id=-1 */
  void* v250 = __runtime_buffer_arg((void*)38400);
  XAie_DmaDesc v251 = __Runtime_dma_bd_config(v1, v43, v250, 3, 512, 2, 0, 0, 0, -1, 1, 1, -1);
  /* DMA BD Config: bd_id=2, len=512, enable_packet=false, packet_id=0, next_bd=3, acquire_lock_id=0, acquire_lock_val=-1, release_lock_id=1, release_lock_val=1, ooo_bd_id=-1 */
  void* v252 = __runtime_buffer_arg((void*)37888);
  XAie_DmaDesc v253 = __Runtime_dma_bd_config(v1, v43, v252, 2, 512, 3, 0, 0, 0, -1, 1, 1, -1);
  /* Create IO: channel_id=0, bd_id=2, tile=(3,4), direction=S2MM */
  io v254 = __Runtime_dma_createio_4(v43, v253, 0, 2, DMA_S2MM);
  ioevent v255 = __Runtime_startio(v1, v254, 1, 1);
  /* DMA BD Config: bd_id=5, len=5120, enable_packet=true, packet_id=5, next_bd=4, acquire_lock_id=5, acquire_lock_val=-1, release_lock_id=4, release_lock_val=1, ooo_bd_id=7 */
  void* v256 = __runtime_buffer_arg((void*)44032);
  XAie_DmaDesc v257 = __Runtime_dma_bd_config(v1, v16, v256, 5, 5120, 4, 1, 5, 5, -1, 4, 1, 7);
  /* DMA BD Config: bd_id=4, len=5120, enable_packet=true, packet_id=5, next_bd=5, acquire_lock_id=5, acquire_lock_val=-1, release_lock_id=4, release_lock_val=1, ooo_bd_id=7 */
  void* v258 = __runtime_buffer_arg((void*)38912);
  XAie_DmaDesc v259 = __Runtime_dma_bd_config(v1, v16, v258, 4, 5120, 5, 1, 5, 5, -1, 4, 1, 7);
  /* Create IO: channel_id=0, bd_id=4, tile=(0,4), direction=MM2S */
  io v260 = __Runtime_dma_createio_4(v16, v259, 0, 4, DMA_MM2S);
  ioevent v261 = __Runtime_startio(v1, v260, 2, 1);
  /* DMA BD Config: bd_id=5, len=5120, enable_packet=true, packet_id=6, next_bd=4, acquire_lock_id=5, acquire_lock_val=-1, release_lock_id=4, release_lock_val=1, ooo_bd_id=8 */
  void* v262 = __runtime_buffer_arg((void*)44032);
  XAie_DmaDesc v263 = __Runtime_dma_bd_config(v1, v23, v262, 5, 5120, 4, 1, 6, 5, -1, 4, 1, 8);
  /* DMA BD Config: bd_id=4, len=5120, enable_packet=true, packet_id=6, next_bd=5, acquire_lock_id=5, acquire_lock_val=-1, release_lock_id=4, release_lock_val=1, ooo_bd_id=8 */
  void* v264 = __runtime_buffer_arg((void*)38912);
  XAie_DmaDesc v265 = __Runtime_dma_bd_config(v1, v23, v264, 4, 5120, 5, 1, 6, 5, -1, 4, 1, 8);
  /* Create IO: channel_id=0, bd_id=4, tile=(1,4), direction=MM2S */
  io v266 = __Runtime_dma_createio_4(v23, v265, 0, 4, DMA_MM2S);
  ioevent v267 = __Runtime_startio(v1, v266, 2, 1);
  /* DMA BD Config: bd_id=5, len=5120, enable_packet=true, packet_id=7, next_bd=4, acquire_lock_id=5, acquire_lock_val=-1, release_lock_id=4, release_lock_val=1, ooo_bd_id=9 */
  void* v268 = __runtime_buffer_arg((void*)44032);
  XAie_DmaDesc v269 = __Runtime_dma_bd_config(v1, v33, v268, 5, 5120, 4, 1, 7, 5, -1, 4, 1, 9);
  /* DMA BD Config: bd_id=4, len=5120, enable_packet=true, packet_id=7, next_bd=5, acquire_lock_id=5, acquire_lock_val=-1, release_lock_id=4, release_lock_val=1, ooo_bd_id=9 */
  void* v270 = __runtime_buffer_arg((void*)38912);
  XAie_DmaDesc v271 = __Runtime_dma_bd_config(v1, v33, v270, 4, 5120, 5, 1, 7, 5, -1, 4, 1, 9);
  /* Create IO: channel_id=0, bd_id=4, tile=(2,4), direction=MM2S */
  io v272 = __Runtime_dma_createio_4(v33, v271, 0, 4, DMA_MM2S);
  ioevent v273 = __Runtime_startio(v1, v272, 2, 1);
  /* DMA BD Config: bd_id=5, len=5120, enable_packet=true, packet_id=8, next_bd=4, acquire_lock_id=5, acquire_lock_val=-1, release_lock_id=4, release_lock_val=1, ooo_bd_id=10 */
  void* v274 = __runtime_buffer_arg((void*)44032);
  XAie_DmaDesc v275 = __Runtime_dma_bd_config(v1, v43, v274, 5, 5120, 4, 1, 8, 5, -1, 4, 1, 10);
  /* DMA BD Config: bd_id=4, len=5120, enable_packet=true, packet_id=8, next_bd=5, acquire_lock_id=5, acquire_lock_val=-1, release_lock_id=4, release_lock_val=1, ooo_bd_id=10 */
  void* v276 = __runtime_buffer_arg((void*)38912);
  XAie_DmaDesc v277 = __Runtime_dma_bd_config(v1, v43, v276, 4, 5120, 5, 1, 8, 5, -1, 4, 1, 10);
  /* Create IO: channel_id=0, bd_id=4, tile=(3,4), direction=MM2S */
  io v278 = __Runtime_dma_createio_4(v43, v277, 0, 4, DMA_MM2S);
  ioevent v279 = __Runtime_startio(v1, v278, 2, 1);
  /* DMA BD Config: bd_id=3, len=512, enable_packet=false, packet_id=0, next_bd=2, acquire_lock_id=0, acquire_lock_val=-1, release_lock_id=1, release_lock_val=1, ooo_bd_id=-1 */
  void* v280 = __runtime_buffer_arg((void*)38400);
  XAie_DmaDesc v281 = __Runtime_dma_bd_config(v1, v17, v280, 3, 512, 2, 0, 0, 0, -1, 1, 1, -1);
  /* DMA BD Config: bd_id=2, len=512, enable_packet=false, packet_id=0, next_bd=3, acquire_lock_id=0, acquire_lock_val=-1, release_lock_id=1, release_lock_val=1, ooo_bd_id=-1 */
  void* v282 = __runtime_buffer_arg((void*)37888);
  XAie_DmaDesc v283 = __Runtime_dma_bd_config(v1, v17, v282, 2, 512, 3, 0, 0, 0, -1, 1, 1, -1);
  /* Create IO: channel_id=0, bd_id=2, tile=(0,5), direction=S2MM */
  io v284 = __Runtime_dma_createio_4(v17, v283, 0, 2, DMA_S2MM);
  ioevent v285 = __Runtime_startio(v1, v284, 1, 1);
  /* DMA BD Config: bd_id=3, len=512, enable_packet=false, packet_id=0, next_bd=2, acquire_lock_id=0, acquire_lock_val=-1, release_lock_id=1, release_lock_val=1, ooo_bd_id=-1 */
  void* v286 = __runtime_buffer_arg((void*)38400);
  XAie_DmaDesc v287 = __Runtime_dma_bd_config(v1, v25, v286, 3, 512, 2, 0, 0, 0, -1, 1, 1, -1);
  /* DMA BD Config: bd_id=2, len=512, enable_packet=false, packet_id=0, next_bd=3, acquire_lock_id=0, acquire_lock_val=-1, release_lock_id=1, release_lock_val=1, ooo_bd_id=-1 */
  void* v288 = __runtime_buffer_arg((void*)37888);
  XAie_DmaDesc v289 = __Runtime_dma_bd_config(v1, v25, v288, 2, 512, 3, 0, 0, 0, -1, 1, 1, -1);
  /* Create IO: channel_id=0, bd_id=2, tile=(1,5), direction=S2MM */
  io v290 = __Runtime_dma_createio_4(v25, v289, 0, 2, DMA_S2MM);
  ioevent v291 = __Runtime_startio(v1, v290, 1, 1);
  /* DMA BD Config: bd_id=3, len=512, enable_packet=false, packet_id=0, next_bd=2, acquire_lock_id=0, acquire_lock_val=-1, release_lock_id=1, release_lock_val=1, ooo_bd_id=-1 */
  void* v292 = __runtime_buffer_arg((void*)38400);
  XAie_DmaDesc v293 = __Runtime_dma_bd_config(v1, v35, v292, 3, 512, 2, 0, 0, 0, -1, 1, 1, -1);
  /* DMA BD Config: bd_id=2, len=512, enable_packet=false, packet_id=0, next_bd=3, acquire_lock_id=0, acquire_lock_val=-1, release_lock_id=1, release_lock_val=1, ooo_bd_id=-1 */
  void* v294 = __runtime_buffer_arg((void*)37888);
  XAie_DmaDesc v295 = __Runtime_dma_bd_config(v1, v35, v294, 2, 512, 3, 0, 0, 0, -1, 1, 1, -1);
  /* Create IO: channel_id=0, bd_id=2, tile=(2,5), direction=S2MM */
  io v296 = __Runtime_dma_createio_4(v35, v295, 0, 2, DMA_S2MM);
  ioevent v297 = __Runtime_startio(v1, v296, 1, 1);
  /* DMA BD Config: bd_id=3, len=512, enable_packet=false, packet_id=0, next_bd=2, acquire_lock_id=0, acquire_lock_val=-1, release_lock_id=1, release_lock_val=1, ooo_bd_id=-1 */
  void* v298 = __runtime_buffer_arg((void*)38400);
  XAie_DmaDesc v299 = __Runtime_dma_bd_config(v1, v45, v298, 3, 512, 2, 0, 0, 0, -1, 1, 1, -1);
  /* DMA BD Config: bd_id=2, len=512, enable_packet=false, packet_id=0, next_bd=3, acquire_lock_id=0, acquire_lock_val=-1, release_lock_id=1, release_lock_val=1, ooo_bd_id=-1 */
  void* v300 = __runtime_buffer_arg((void*)37888);
  XAie_DmaDesc v301 = __Runtime_dma_bd_config(v1, v45, v300, 2, 512, 3, 0, 0, 0, -1, 1, 1, -1);
  /* Create IO: channel_id=0, bd_id=2, tile=(3,5), direction=S2MM */
  io v302 = __Runtime_dma_createio_4(v45, v301, 0, 2, DMA_S2MM);
  ioevent v303 = __Runtime_startio(v1, v302, 1, 1);
  /* DMA BD Config: bd_id=5, len=5120, enable_packet=true, packet_id=9, next_bd=4, acquire_lock_id=5, acquire_lock_val=-1, release_lock_id=4, release_lock_val=1, ooo_bd_id=1 */
  void* v304 = __runtime_buffer_arg((void*)44032);
  XAie_DmaDesc v305 = __Runtime_dma_bd_config(v1, v17, v304, 5, 5120, 4, 1, 9, 5, -1, 4, 1, 1);
  /* DMA BD Config: bd_id=4, len=5120, enable_packet=true, packet_id=9, next_bd=5, acquire_lock_id=5, acquire_lock_val=-1, release_lock_id=4, release_lock_val=1, ooo_bd_id=1 */
  void* v306 = __runtime_buffer_arg((void*)38912);
  XAie_DmaDesc v307 = __Runtime_dma_bd_config(v1, v17, v306, 4, 5120, 5, 1, 9, 5, -1, 4, 1, 1);
  /* Create IO: channel_id=0, bd_id=4, tile=(0,5), direction=MM2S */
  io v308 = __Runtime_dma_createio_4(v17, v307, 0, 4, DMA_MM2S);
  ioevent v309 = __Runtime_startio(v1, v308, 2, 1);
  /* DMA BD Config: bd_id=5, len=5120, enable_packet=true, packet_id=10, next_bd=4, acquire_lock_id=5, acquire_lock_val=-1, release_lock_id=4, release_lock_val=1, ooo_bd_id=2 */
  void* v310 = __runtime_buffer_arg((void*)44032);
  XAie_DmaDesc v311 = __Runtime_dma_bd_config(v1, v25, v310, 5, 5120, 4, 1, 10, 5, -1, 4, 1, 2);
  /* DMA BD Config: bd_id=4, len=5120, enable_packet=true, packet_id=10, next_bd=5, acquire_lock_id=5, acquire_lock_val=-1, release_lock_id=4, release_lock_val=1, ooo_bd_id=2 */
  void* v312 = __runtime_buffer_arg((void*)38912);
  XAie_DmaDesc v313 = __Runtime_dma_bd_config(v1, v25, v312, 4, 5120, 5, 1, 10, 5, -1, 4, 1, 2);
  /* Create IO: channel_id=0, bd_id=4, tile=(1,5), direction=MM2S */
  io v314 = __Runtime_dma_createio_4(v25, v313, 0, 4, DMA_MM2S);
  ioevent v315 = __Runtime_startio(v1, v314, 2, 1);
  /* DMA BD Config: bd_id=5, len=5120, enable_packet=true, packet_id=11, next_bd=4, acquire_lock_id=5, acquire_lock_val=-1, release_lock_id=4, release_lock_val=1, ooo_bd_id=3 */
  void* v316 = __runtime_buffer_arg((void*)44032);
  XAie_DmaDesc v317 = __Runtime_dma_bd_config(v1, v35, v316, 5, 5120, 4, 1, 11, 5, -1, 4, 1, 3);
  /* DMA BD Config: bd_id=4, len=5120, enable_packet=true, packet_id=11, next_bd=5, acquire_lock_id=5, acquire_lock_val=-1, release_lock_id=4, release_lock_val=1, ooo_bd_id=3 */
  void* v318 = __runtime_buffer_arg((void*)38912);
  XAie_DmaDesc v319 = __Runtime_dma_bd_config(v1, v35, v318, 4, 5120, 5, 1, 11, 5, -1, 4, 1, 3);
  /* Create IO: channel_id=0, bd_id=4, tile=(2,5), direction=MM2S */
  io v320 = __Runtime_dma_createio_4(v35, v319, 0, 4, DMA_MM2S);
  ioevent v321 = __Runtime_startio(v1, v320, 2, 1);
  /* DMA BD Config: bd_id=5, len=5120, enable_packet=true, packet_id=12, next_bd=4, acquire_lock_id=5, acquire_lock_val=-1, release_lock_id=4, release_lock_val=1, ooo_bd_id=4 */
  void* v322 = __runtime_buffer_arg((void*)44032);
  XAie_DmaDesc v323 = __Runtime_dma_bd_config(v1, v45, v322, 5, 5120, 4, 1, 12, 5, -1, 4, 1, 4);
  /* DMA BD Config: bd_id=4, len=5120, enable_packet=true, packet_id=12, next_bd=5, acquire_lock_id=5, acquire_lock_val=-1, release_lock_id=4, release_lock_val=1, ooo_bd_id=4 */
  void* v324 = __runtime_buffer_arg((void*)38912);
  XAie_DmaDesc v325 = __Runtime_dma_bd_config(v1, v45, v324, 4, 5120, 5, 1, 12, 5, -1, 4, 1, 4);
  /* Create IO: channel_id=0, bd_id=4, tile=(3,5), direction=MM2S */
  io v326 = __Runtime_dma_createio_4(v45, v325, 0, 4, DMA_MM2S);
  ioevent v327 = __Runtime_startio(v1, v326, 2, 1);
  /* DMA BD Config: bd_id=3, len=512, enable_packet=false, packet_id=0, next_bd=2, acquire_lock_id=0, acquire_lock_val=-1, release_lock_id=1, release_lock_val=1, ooo_bd_id=-1 */
  void* v328 = __runtime_buffer_arg((void*)38400);
  XAie_DmaDesc v329 = __Runtime_dma_bd_config(v1, v18, v328, 3, 512, 2, 0, 0, 0, -1, 1, 1, -1);
  /* DMA BD Config: bd_id=2, len=512, enable_packet=false, packet_id=0, next_bd=3, acquire_lock_id=0, acquire_lock_val=-1, release_lock_id=1, release_lock_val=1, ooo_bd_id=-1 */
  void* v330 = __runtime_buffer_arg((void*)37888);
  XAie_DmaDesc v331 = __Runtime_dma_bd_config(v1, v18, v330, 2, 512, 3, 0, 0, 0, -1, 1, 1, -1);
  /* Create IO: channel_id=0, bd_id=2, tile=(0,6), direction=S2MM */
  io v332 = __Runtime_dma_createio_4(v18, v331, 0, 2, DMA_S2MM);
  ioevent v333 = __Runtime_startio(v1, v332, 1, 1);
  /* DMA BD Config: bd_id=3, len=512, enable_packet=false, packet_id=0, next_bd=2, acquire_lock_id=0, acquire_lock_val=-1, release_lock_id=1, release_lock_val=1, ooo_bd_id=-1 */
  void* v334 = __runtime_buffer_arg((void*)38400);
  XAie_DmaDesc v335 = __Runtime_dma_bd_config(v1, v27, v334, 3, 512, 2, 0, 0, 0, -1, 1, 1, -1);
  /* DMA BD Config: bd_id=2, len=512, enable_packet=false, packet_id=0, next_bd=3, acquire_lock_id=0, acquire_lock_val=-1, release_lock_id=1, release_lock_val=1, ooo_bd_id=-1 */
  void* v336 = __runtime_buffer_arg((void*)37888);
  XAie_DmaDesc v337 = __Runtime_dma_bd_config(v1, v27, v336, 2, 512, 3, 0, 0, 0, -1, 1, 1, -1);
  /* Create IO: channel_id=0, bd_id=2, tile=(1,6), direction=S2MM */
  io v338 = __Runtime_dma_createio_4(v27, v337, 0, 2, DMA_S2MM);
  ioevent v339 = __Runtime_startio(v1, v338, 1, 1);
  /* DMA BD Config: bd_id=3, len=512, enable_packet=false, packet_id=0, next_bd=2, acquire_lock_id=0, acquire_lock_val=-1, release_lock_id=1, release_lock_val=1, ooo_bd_id=-1 */
  void* v340 = __runtime_buffer_arg((void*)38400);
  XAie_DmaDesc v341 = __Runtime_dma_bd_config(v1, v37, v340, 3, 512, 2, 0, 0, 0, -1, 1, 1, -1);
  /* DMA BD Config: bd_id=2, len=512, enable_packet=false, packet_id=0, next_bd=3, acquire_lock_id=0, acquire_lock_val=-1, release_lock_id=1, release_lock_val=1, ooo_bd_id=-1 */
  void* v342 = __runtime_buffer_arg((void*)37888);
  XAie_DmaDesc v343 = __Runtime_dma_bd_config(v1, v37, v342, 2, 512, 3, 0, 0, 0, -1, 1, 1, -1);
  /* Create IO: channel_id=0, bd_id=2, tile=(2,6), direction=S2MM */
  io v344 = __Runtime_dma_createio_4(v37, v343, 0, 2, DMA_S2MM);
  ioevent v345 = __Runtime_startio(v1, v344, 1, 1);
  /* DMA BD Config: bd_id=3, len=512, enable_packet=false, packet_id=0, next_bd=2, acquire_lock_id=0, acquire_lock_val=-1, release_lock_id=1, release_lock_val=1, ooo_bd_id=-1 */
  void* v346 = __runtime_buffer_arg((void*)38400);
  XAie_DmaDesc v347 = __Runtime_dma_bd_config(v1, v47, v346, 3, 512, 2, 0, 0, 0, -1, 1, 1, -1);
  /* DMA BD Config: bd_id=2, len=512, enable_packet=false, packet_id=0, next_bd=3, acquire_lock_id=0, acquire_lock_val=-1, release_lock_id=1, release_lock_val=1, ooo_bd_id=-1 */
  void* v348 = __runtime_buffer_arg((void*)37888);
  XAie_DmaDesc v349 = __Runtime_dma_bd_config(v1, v47, v348, 2, 512, 3, 0, 0, 0, -1, 1, 1, -1);
  /* Create IO: channel_id=0, bd_id=2, tile=(3,6), direction=S2MM */
  io v350 = __Runtime_dma_createio_4(v47, v349, 0, 2, DMA_S2MM);
  ioevent v351 = __Runtime_startio(v1, v350, 1, 1);
  /* DMA BD Config: bd_id=5, len=5120, enable_packet=true, packet_id=13, next_bd=4, acquire_lock_id=5, acquire_lock_val=-1, release_lock_id=4, release_lock_val=1, ooo_bd_id=6 */
  void* v352 = __runtime_buffer_arg((void*)44032);
  XAie_DmaDesc v353 = __Runtime_dma_bd_config(v1, v18, v352, 5, 5120, 4, 1, 13, 5, -1, 4, 1, 6);
  /* DMA BD Config: bd_id=4, len=5120, enable_packet=true, packet_id=13, next_bd=5, acquire_lock_id=5, acquire_lock_val=-1, release_lock_id=4, release_lock_val=1, ooo_bd_id=6 */
  void* v354 = __runtime_buffer_arg((void*)38912);
  XAie_DmaDesc v355 = __Runtime_dma_bd_config(v1, v18, v354, 4, 5120, 5, 1, 13, 5, -1, 4, 1, 6);
  /* Create IO: channel_id=0, bd_id=4, tile=(0,6), direction=MM2S */
  io v356 = __Runtime_dma_createio_4(v18, v355, 0, 4, DMA_MM2S);
  ioevent v357 = __Runtime_startio(v1, v356, 2, 1);
  /* DMA BD Config: bd_id=5, len=5120, enable_packet=true, packet_id=14, next_bd=4, acquire_lock_id=5, acquire_lock_val=-1, release_lock_id=4, release_lock_val=1, ooo_bd_id=7 */
  void* v358 = __runtime_buffer_arg((void*)44032);
  XAie_DmaDesc v359 = __Runtime_dma_bd_config(v1, v27, v358, 5, 5120, 4, 1, 14, 5, -1, 4, 1, 7);
  /* DMA BD Config: bd_id=4, len=5120, enable_packet=true, packet_id=14, next_bd=5, acquire_lock_id=5, acquire_lock_val=-1, release_lock_id=4, release_lock_val=1, ooo_bd_id=7 */
  void* v360 = __runtime_buffer_arg((void*)38912);
  XAie_DmaDesc v361 = __Runtime_dma_bd_config(v1, v27, v360, 4, 5120, 5, 1, 14, 5, -1, 4, 1, 7);
  /* Create IO: channel_id=0, bd_id=4, tile=(1,6), direction=MM2S */
  io v362 = __Runtime_dma_createio_4(v27, v361, 0, 4, DMA_MM2S);
  ioevent v363 = __Runtime_startio(v1, v362, 2, 1);
  /* DMA BD Config: bd_id=5, len=5120, enable_packet=true, packet_id=15, next_bd=4, acquire_lock_id=5, acquire_lock_val=-1, release_lock_id=4, release_lock_val=1, ooo_bd_id=8 */
  void* v364 = __runtime_buffer_arg((void*)44032);
  XAie_DmaDesc v365 = __Runtime_dma_bd_config(v1, v37, v364, 5, 5120, 4, 1, 15, 5, -1, 4, 1, 8);
  /* DMA BD Config: bd_id=4, len=5120, enable_packet=true, packet_id=15, next_bd=5, acquire_lock_id=5, acquire_lock_val=-1, release_lock_id=4, release_lock_val=1, ooo_bd_id=8 */
  void* v366 = __runtime_buffer_arg((void*)38912);
  XAie_DmaDesc v367 = __Runtime_dma_bd_config(v1, v37, v366, 4, 5120, 5, 1, 15, 5, -1, 4, 1, 8);
  /* Create IO: channel_id=0, bd_id=4, tile=(2,6), direction=MM2S */
  io v368 = __Runtime_dma_createio_4(v37, v367, 0, 4, DMA_MM2S);
  ioevent v369 = __Runtime_startio(v1, v368, 2, 1);
  /* DMA BD Config: bd_id=5, len=5120, enable_packet=true, packet_id=16, next_bd=4, acquire_lock_id=5, acquire_lock_val=-1, release_lock_id=4, release_lock_val=1, ooo_bd_id=9 */
  void* v370 = __runtime_buffer_arg((void*)44032);
  XAie_DmaDesc v371 = __Runtime_dma_bd_config(v1, v47, v370, 5, 5120, 4, 1, 16, 5, -1, 4, 1, 9);
  /* DMA BD Config: bd_id=4, len=5120, enable_packet=true, packet_id=16, next_bd=5, acquire_lock_id=5, acquire_lock_val=-1, release_lock_id=4, release_lock_val=1, ooo_bd_id=9 */
  void* v372 = __runtime_buffer_arg((void*)38912);
  XAie_DmaDesc v373 = __Runtime_dma_bd_config(v1, v47, v372, 4, 5120, 5, 1, 16, 5, -1, 4, 1, 9);
  /* Create IO: channel_id=0, bd_id=4, tile=(3,6), direction=MM2S */
  io v374 = __Runtime_dma_createio_4(v47, v373, 0, 4, DMA_MM2S);
  ioevent v375 = __Runtime_startio(v1, v374, 2, 1);
  for (size_t v376 = v7; v376 < v6; v376 += v5) {
    /* Allocated BD ID 2 for tile (0,0) */;
    /* DMA BD Config: bd_id=0, len=51200, enable_packet=false, packet_id=0, next_bd=-1, acquire_lock_id=0, acquire_lock_val=0, release_lock_id=0, release_lock_val=0, ooo_bd_id=-1, num_dims=3, d0_stride=4, d0_wrap=8, d1_stride=640, d1_wrap=80, d2_stride=32, d2_wrap=20 */;
    void* v377 = __runtime_buffer_arg(v13);
    XAie_DmaDesc v378 = __Runtime_dma_bd_config_multidim(v1, v14, v377, 0, 51200, -1, 0, 0, 0, 0, 0, 0, -1, 3, 4, 8, 640, 80, 32, 20, 0, 0);
    /* Create IO: channel_id=0, bd_id=0, tile=(0,0), direction=MM2S */;
    io v379 = __Runtime_dma_createio_4(v14, v378, 0, 0, DMA_MM2S);
    ioevent v380 = __Runtime_startio(v1, v379, 2, 1);
    /* Allocated BD ID 2 for tile (1,0) */;
    /* DMA BD Config: bd_id=0, len=51200, enable_packet=false, packet_id=0, next_bd=-1, acquire_lock_id=0, acquire_lock_val=0, release_lock_id=0, release_lock_val=0, ooo_bd_id=-1, num_dims=3, d0_stride=4, d0_wrap=8, d1_stride=640, d1_wrap=80, d2_stride=32, d2_wrap=20 */;
    void* v381 = __runtime_buffer_arg(v19);
    XAie_DmaDesc v382 = __Runtime_dma_bd_config_multidim(v1, v20, v381, 0, 51200, -1, 0, 0, 0, 0, 0, 0, -1, 3, 4, 8, 640, 80, 32, 20, 0, 0);
    /* Create IO: channel_id=0, bd_id=0, tile=(1,0), direction=MM2S */;
    io v383 = __Runtime_dma_createio_4(v20, v382, 0, 0, DMA_MM2S);
    ioevent v384 = __Runtime_startio(v1, v383, 2, 1);
    /* Allocated BD ID 2 for tile (2,0) */;
    /* DMA BD Config: bd_id=0, len=51200, enable_packet=false, packet_id=0, next_bd=-1, acquire_lock_id=0, acquire_lock_val=0, release_lock_id=0, release_lock_val=0, ooo_bd_id=-1, num_dims=3, d0_stride=4, d0_wrap=8, d1_stride=640, d1_wrap=80, d2_stride=32, d2_wrap=20 */;
    void* v385 = __runtime_buffer_arg(v29);
    XAie_DmaDesc v386 = __Runtime_dma_bd_config_multidim(v1, v30, v385, 0, 51200, -1, 0, 0, 0, 0, 0, 0, -1, 3, 4, 8, 640, 80, 32, 20, 0, 0);
    /* Create IO: channel_id=0, bd_id=0, tile=(2,0), direction=MM2S */;
    io v387 = __Runtime_dma_createio_4(v30, v386, 0, 0, DMA_MM2S);
    ioevent v388 = __Runtime_startio(v1, v387, 2, 1);
    /* Allocated BD ID 4 for tile (3,0) */;
    /* DMA BD Config: bd_id=0, len=51200, enable_packet=false, packet_id=0, next_bd=-1, acquire_lock_id=0, acquire_lock_val=0, release_lock_id=0, release_lock_val=0, ooo_bd_id=-1, num_dims=3, d0_stride=4, d0_wrap=8, d1_stride=640, d1_wrap=80, d2_stride=32, d2_wrap=20 */;
    void* v389 = __runtime_buffer_arg(v39);
    XAie_DmaDesc v390 = __Runtime_dma_bd_config_multidim(v1, v40, v389, 0, 51200, -1, 0, 0, 0, 0, 0, 0, -1, 3, 4, 8, 640, 80, 32, 20, 0, 0);
    /* Create IO: channel_id=0, bd_id=0, tile=(3,0), direction=MM2S */;
    io v391 = __Runtime_dma_createio_4(v40, v390, 0, 0, DMA_MM2S);
    ioevent v392 = __Runtime_startio(v1, v391, 4, 1);
    int32_t v393 = (int32_t) v376;
    int32_t v394 = v393 * v8;
    /* Allocated BD ID 3 for tile (0,0) */;
    /* DMA BD Config: bd_id=1, len=10240, enable_packet=false, packet_id=0, next_bd=-1, acquire_lock_id=0, acquire_lock_val=0, release_lock_id=0, release_lock_val=0, ooo_bd_id=-1, num_dims=3, d0_stride=4, d0_wrap=8, d1_stride=640, d1_wrap=16, d2_stride=32, d2_wrap=20 */;
    void* v395 = __runtime_buffer_arg(v49);
    int64_t v396 = (int64_t) v394;
    void* v397 = __runtime_buffer_offset(v395, v396);
    XAie_DmaDesc v398 = __Runtime_dma_bd_config_multidim(v1, v14, v397, 1, 10240, -1, 0, 0, 0, 0, 0, 0, -1, 3, 4, 8, 640, 16, 32, 20, 0, 0);
    /* Create IO: channel_id=1, bd_id=1, tile=(0,0), direction=MM2S */;
    io v399 = __Runtime_dma_createio_4(v14, v398, 1, 1, DMA_MM2S);
    ioevent v400 = __Runtime_startio(v1, v399, 3, 1);
    int32_t v401 = (int32_t) v376;
    int32_t v402 = v401 * v9;
    int32_t v403 = v402 + v10;
    int32_t v404 = v402 + v11;
    int32_t v405 = v402 + v12;
    /* Allocated BD ID 5 for tile (3,0) */;
    /* DMA BD Config: bd_id=5, len=5120, enable_packet=false, packet_id=4, next_bd=-1, acquire_lock_id=-1, acquire_lock_val=0, release_lock_id=-1, release_lock_val=0, ooo_bd_id=-1, num_dims=2, d0_stride=4, d0_wrap=80, d1_stride=1280, d1_wrap=16 */;
    void* v406 = __runtime_buffer_arg(v50);
    int64_t v407 = (int64_t) v403;
    void* v408 = __runtime_buffer_offset(v406, v407);
    XAie_DmaDesc v409 = __Runtime_dma_bd_config_multidim(v1, v40, v408, 5, 5120, -1, 0, 4, -1, 0, -1, 0, -1, 2, 4, 80, 1280, 16, 0, 0, 0, 0);
    /* DMA BD Config: bd_id=4, len=5120, enable_packet=false, packet_id=3, next_bd=-1, acquire_lock_id=-1, acquire_lock_val=0, release_lock_id=-1, release_lock_val=0, ooo_bd_id=-1, num_dims=2, d0_stride=4, d0_wrap=80, d1_stride=1280, d1_wrap=16 */;
    void* v410 = __runtime_buffer_arg(v50);
    int64_t v411 = (int64_t) v404;
    void* v412 = __runtime_buffer_offset(v410, v411);
    XAie_DmaDesc v413 = __Runtime_dma_bd_config_multidim(v1, v40, v412, 4, 5120, -1, 0, 3, -1, 0, -1, 0, -1, 2, 4, 80, 1280, 16, 0, 0, 0, 0);
    /* DMA BD Config: bd_id=3, len=5120, enable_packet=false, packet_id=2, next_bd=-1, acquire_lock_id=-1, acquire_lock_val=0, release_lock_id=-1, release_lock_val=0, ooo_bd_id=-1, num_dims=2, d0_stride=4, d0_wrap=80, d1_stride=1280, d1_wrap=16 */;
    void* v414 = __runtime_buffer_arg(v50);
    int64_t v415 = (int64_t) v405;
    void* v416 = __runtime_buffer_offset(v414, v415);
    XAie_DmaDesc v417 = __Runtime_dma_bd_config_multidim(v1, v40, v416, 3, 5120, -1, 0, 2, -1, 0, -1, 0, -1, 2, 4, 80, 1280, 16, 0, 0, 0, 0);
    /* DMA BD Config: bd_id=2, len=5120, enable_packet=false, packet_id=1, next_bd=-1, acquire_lock_id=-1, acquire_lock_val=0, release_lock_id=-1, release_lock_val=0, ooo_bd_id=-1, num_dims=2, d0_stride=4, d0_wrap=80, d1_stride=1280, d1_wrap=16 */;
    void* v418 = __runtime_buffer_arg(v50);
    int64_t v419 = (int64_t) v402;
    void* v420 = __runtime_buffer_offset(v418, v419);
    XAie_DmaDesc v421 = __Runtime_dma_bd_config_multidim(v1, v40, v420, 2, 5120, -1, 0, 1, -1, 0, -1, 0, -1, 2, 4, 80, 1280, 16, 0, 0, 0, 0);
    /* Create IO: channel_id=0, bd_id=2, tile=(3,0), direction=S2MM */;
    /* Enable out-of-order BD on tile(3,0) ch=0 dir=S2MM */;
    __Runtime_dma_channel_enable_ooo(v1, v40, 0, DMA_S2MM);
    io v422 = __Runtime_dma_createio_4(v40, v421, 0, 2, DMA_S2MM);
    ioevent v423 = __Runtime_startio(v1, v422, 5, 4);
    int32_t v424 = (int32_t) v376;
    int32_t v425 = v424 * v8;
    /* Allocated BD ID 3 for tile (1,0) */;
    /* DMA BD Config: bd_id=1, len=10240, enable_packet=false, packet_id=0, next_bd=-1, acquire_lock_id=0, acquire_lock_val=0, release_lock_id=0, release_lock_val=0, ooo_bd_id=-1, num_dims=3, d0_stride=4, d0_wrap=8, d1_stride=640, d1_wrap=16, d2_stride=32, d2_wrap=20 */;
    void* v426 = __runtime_buffer_arg(v55);
    int64_t v427 = (int64_t) v425;
    void* v428 = __runtime_buffer_offset(v426, v427);
    XAie_DmaDesc v429 = __Runtime_dma_bd_config_multidim(v1, v20, v428, 1, 10240, -1, 0, 0, 0, 0, 0, 0, -1, 3, 4, 8, 640, 16, 32, 20, 0, 0);
    /* Create IO: channel_id=1, bd_id=1, tile=(1,0), direction=MM2S */;
    io v430 = __Runtime_dma_createio_4(v20, v429, 1, 1, DMA_MM2S);
    ioevent v431 = __Runtime_startio(v1, v430, 3, 1);
    int32_t v432 = (int32_t) v376;
    int32_t v433 = v432 * v9;
    int32_t v434 = v433 + v10;
    int32_t v435 = v433 + v11;
    int32_t v436 = v433 + v12;
    /* Allocated BD ID 6 for tile (3,0) */;
    /* DMA BD Config: bd_id=10, len=5120, enable_packet=false, packet_id=8, next_bd=-1, acquire_lock_id=-1, acquire_lock_val=0, release_lock_id=-1, release_lock_val=0, ooo_bd_id=-1, num_dims=2, d0_stride=4, d0_wrap=80, d1_stride=1280, d1_wrap=16 */;
    void* v437 = __runtime_buffer_arg(v60);
    int64_t v438 = (int64_t) v434;
    void* v439 = __runtime_buffer_offset(v437, v438);
    XAie_DmaDesc v440 = __Runtime_dma_bd_config_multidim(v1, v40, v439, 10, 5120, -1, 0, 8, -1, 0, -1, 0, -1, 2, 4, 80, 1280, 16, 0, 0, 0, 0);
    /* DMA BD Config: bd_id=9, len=5120, enable_packet=false, packet_id=7, next_bd=-1, acquire_lock_id=-1, acquire_lock_val=0, release_lock_id=-1, release_lock_val=0, ooo_bd_id=-1, num_dims=2, d0_stride=4, d0_wrap=80, d1_stride=1280, d1_wrap=16 */;
    void* v441 = __runtime_buffer_arg(v60);
    int64_t v442 = (int64_t) v435;
    void* v443 = __runtime_buffer_offset(v441, v442);
    XAie_DmaDesc v444 = __Runtime_dma_bd_config_multidim(v1, v40, v443, 9, 5120, -1, 0, 7, -1, 0, -1, 0, -1, 2, 4, 80, 1280, 16, 0, 0, 0, 0);
    /* DMA BD Config: bd_id=8, len=5120, enable_packet=false, packet_id=6, next_bd=-1, acquire_lock_id=-1, acquire_lock_val=0, release_lock_id=-1, release_lock_val=0, ooo_bd_id=-1, num_dims=2, d0_stride=4, d0_wrap=80, d1_stride=1280, d1_wrap=16 */;
    void* v445 = __runtime_buffer_arg(v60);
    int64_t v446 = (int64_t) v436;
    void* v447 = __runtime_buffer_offset(v445, v446);
    XAie_DmaDesc v448 = __Runtime_dma_bd_config_multidim(v1, v40, v447, 8, 5120, -1, 0, 6, -1, 0, -1, 0, -1, 2, 4, 80, 1280, 16, 0, 0, 0, 0);
    /* DMA BD Config: bd_id=7, len=5120, enable_packet=false, packet_id=5, next_bd=-1, acquire_lock_id=-1, acquire_lock_val=0, release_lock_id=-1, release_lock_val=0, ooo_bd_id=-1, num_dims=2, d0_stride=4, d0_wrap=80, d1_stride=1280, d1_wrap=16 */;
    void* v449 = __runtime_buffer_arg(v60);
    int64_t v450 = (int64_t) v433;
    void* v451 = __runtime_buffer_offset(v449, v450);
    XAie_DmaDesc v452 = __Runtime_dma_bd_config_multidim(v1, v40, v451, 7, 5120, -1, 0, 5, -1, 0, -1, 0, -1, 2, 4, 80, 1280, 16, 0, 0, 0, 0);
    /* Create IO: channel_id=1, bd_id=7, tile=(3,0), direction=S2MM */;
    /* Enable out-of-order BD on tile(3,0) ch=1 dir=S2MM */;
    __Runtime_dma_channel_enable_ooo(v1, v40, 1, DMA_S2MM);
    io v453 = __Runtime_dma_createio_4(v40, v452, 1, 7, DMA_S2MM);
    ioevent v454 = __Runtime_startio(v1, v453, 6, 4);
    int32_t v455 = (int32_t) v376;
    int32_t v456 = v455 * v8;
    /* Allocated BD ID 3 for tile (2,0) */;
    /* DMA BD Config: bd_id=1, len=10240, enable_packet=false, packet_id=0, next_bd=-1, acquire_lock_id=0, acquire_lock_val=0, release_lock_id=0, release_lock_val=0, ooo_bd_id=-1, num_dims=3, d0_stride=4, d0_wrap=8, d1_stride=640, d1_wrap=16, d2_stride=32, d2_wrap=20 */;
    void* v457 = __runtime_buffer_arg(v65);
    int64_t v458 = (int64_t) v456;
    void* v459 = __runtime_buffer_offset(v457, v458);
    XAie_DmaDesc v460 = __Runtime_dma_bd_config_multidim(v1, v30, v459, 1, 10240, -1, 0, 0, 0, 0, 0, 0, -1, 3, 4, 8, 640, 16, 32, 20, 0, 0);
    /* Create IO: channel_id=1, bd_id=1, tile=(2,0), direction=MM2S */;
    io v461 = __Runtime_dma_createio_4(v30, v460, 1, 1, DMA_MM2S);
    ioevent v462 = __Runtime_startio(v1, v461, 3, 1);
    int32_t v463 = (int32_t) v376;
    int32_t v464 = v463 * v9;
    int32_t v465 = v464 + v10;
    int32_t v466 = v464 + v11;
    int32_t v467 = v464 + v12;
    /* Allocated BD ID 2 for tile (4,0) */;
    /* DMA BD Config: bd_id=4, len=5120, enable_packet=false, packet_id=12, next_bd=-1, acquire_lock_id=-1, acquire_lock_val=0, release_lock_id=-1, release_lock_val=0, ooo_bd_id=-1, num_dims=2, d0_stride=4, d0_wrap=80, d1_stride=1280, d1_wrap=16 */;
    void* v468 = __runtime_buffer_arg(v70);
    int64_t v469 = (int64_t) v465;
    void* v470 = __runtime_buffer_offset(v468, v469);
    XAie_DmaDesc v471 = __Runtime_dma_bd_config_multidim(v1, v71, v470, 4, 5120, -1, 0, 12, -1, 0, -1, 0, -1, 2, 4, 80, 1280, 16, 0, 0, 0, 0);
    /* DMA BD Config: bd_id=3, len=5120, enable_packet=false, packet_id=11, next_bd=-1, acquire_lock_id=-1, acquire_lock_val=0, release_lock_id=-1, release_lock_val=0, ooo_bd_id=-1, num_dims=2, d0_stride=4, d0_wrap=80, d1_stride=1280, d1_wrap=16 */;
    void* v472 = __runtime_buffer_arg(v70);
    int64_t v473 = (int64_t) v466;
    void* v474 = __runtime_buffer_offset(v472, v473);
    XAie_DmaDesc v475 = __Runtime_dma_bd_config_multidim(v1, v71, v474, 3, 5120, -1, 0, 11, -1, 0, -1, 0, -1, 2, 4, 80, 1280, 16, 0, 0, 0, 0);
    /* DMA BD Config: bd_id=2, len=5120, enable_packet=false, packet_id=10, next_bd=-1, acquire_lock_id=-1, acquire_lock_val=0, release_lock_id=-1, release_lock_val=0, ooo_bd_id=-1, num_dims=2, d0_stride=4, d0_wrap=80, d1_stride=1280, d1_wrap=16 */;
    void* v476 = __runtime_buffer_arg(v70);
    int64_t v477 = (int64_t) v467;
    void* v478 = __runtime_buffer_offset(v476, v477);
    XAie_DmaDesc v479 = __Runtime_dma_bd_config_multidim(v1, v71, v478, 2, 5120, -1, 0, 10, -1, 0, -1, 0, -1, 2, 4, 80, 1280, 16, 0, 0, 0, 0);
    /* DMA BD Config: bd_id=1, len=5120, enable_packet=false, packet_id=9, next_bd=-1, acquire_lock_id=-1, acquire_lock_val=0, release_lock_id=-1, release_lock_val=0, ooo_bd_id=-1, num_dims=2, d0_stride=4, d0_wrap=80, d1_stride=1280, d1_wrap=16 */;
    void* v480 = __runtime_buffer_arg(v70);
    int64_t v481 = (int64_t) v464;
    void* v482 = __runtime_buffer_offset(v480, v481);
    XAie_DmaDesc v483 = __Runtime_dma_bd_config_multidim(v1, v71, v482, 1, 5120, -1, 0, 9, -1, 0, -1, 0, -1, 2, 4, 80, 1280, 16, 0, 0, 0, 0);
    /* Create IO: channel_id=0, bd_id=1, tile=(4,0), direction=S2MM */;
    /* Enable out-of-order BD on tile(4,0) ch=0 dir=S2MM */;
    __Runtime_dma_channel_enable_ooo(v1, v71, 0, DMA_S2MM);
    io v484 = __Runtime_dma_createio_4(v71, v483, 0, 1, DMA_S2MM);
    ioevent v485 = __Runtime_startio(v1, v484, 2, 4);
    int32_t v486 = (int32_t) v376;
    int32_t v487 = v486 * v8;
    /* Allocated BD ID 7 for tile (3,0) */;
    /* DMA BD Config: bd_id=11, len=10240, enable_packet=false, packet_id=0, next_bd=-1, acquire_lock_id=0, acquire_lock_val=0, release_lock_id=0, release_lock_val=0, ooo_bd_id=-1, num_dims=3, d0_stride=4, d0_wrap=8, d1_stride=640, d1_wrap=16, d2_stride=32, d2_wrap=20 */;
    void* v488 = __runtime_buffer_arg(v76);
    int64_t v489 = (int64_t) v487;
    void* v490 = __runtime_buffer_offset(v488, v489);
    XAie_DmaDesc v491 = __Runtime_dma_bd_config_multidim(v1, v40, v490, 11, 10240, -1, 0, 0, 0, 0, 0, 0, -1, 3, 4, 8, 640, 16, 32, 20, 0, 0);
    /* Create IO: channel_id=1, bd_id=11, tile=(3,0), direction=MM2S */;
    io v492 = __Runtime_dma_createio_4(v40, v491, 1, 11, DMA_MM2S);
    ioevent v493 = __Runtime_startio(v1, v492, 7, 1);
    int32_t v494 = (int32_t) v376;
    int32_t v495 = v494 * v9;
    int32_t v496 = v495 + v10;
    int32_t v497 = v495 + v11;
    int32_t v498 = v495 + v12;
    /* Allocated BD ID 3 for tile (4,0) */;
    /* DMA BD Config: bd_id=9, len=5120, enable_packet=false, packet_id=16, next_bd=-1, acquire_lock_id=-1, acquire_lock_val=0, release_lock_id=-1, release_lock_val=0, ooo_bd_id=-1, num_dims=2, d0_stride=4, d0_wrap=80, d1_stride=1280, d1_wrap=16 */;
    void* v499 = __runtime_buffer_arg(v81);
    int64_t v500 = (int64_t) v496;
    void* v501 = __runtime_buffer_offset(v499, v500);
    XAie_DmaDesc v502 = __Runtime_dma_bd_config_multidim(v1, v71, v501, 9, 5120, -1, 0, 16, -1, 0, -1, 0, -1, 2, 4, 80, 1280, 16, 0, 0, 0, 0);
    /* DMA BD Config: bd_id=8, len=5120, enable_packet=false, packet_id=15, next_bd=-1, acquire_lock_id=-1, acquire_lock_val=0, release_lock_id=-1, release_lock_val=0, ooo_bd_id=-1, num_dims=2, d0_stride=4, d0_wrap=80, d1_stride=1280, d1_wrap=16 */;
    void* v503 = __runtime_buffer_arg(v81);
    int64_t v504 = (int64_t) v497;
    void* v505 = __runtime_buffer_offset(v503, v504);
    XAie_DmaDesc v506 = __Runtime_dma_bd_config_multidim(v1, v71, v505, 8, 5120, -1, 0, 15, -1, 0, -1, 0, -1, 2, 4, 80, 1280, 16, 0, 0, 0, 0);
    /* DMA BD Config: bd_id=7, len=5120, enable_packet=false, packet_id=14, next_bd=-1, acquire_lock_id=-1, acquire_lock_val=0, release_lock_id=-1, release_lock_val=0, ooo_bd_id=-1, num_dims=2, d0_stride=4, d0_wrap=80, d1_stride=1280, d1_wrap=16 */;
    void* v507 = __runtime_buffer_arg(v81);
    int64_t v508 = (int64_t) v498;
    void* v509 = __runtime_buffer_offset(v507, v508);
    XAie_DmaDesc v510 = __Runtime_dma_bd_config_multidim(v1, v71, v509, 7, 5120, -1, 0, 14, -1, 0, -1, 0, -1, 2, 4, 80, 1280, 16, 0, 0, 0, 0);
    /* DMA BD Config: bd_id=6, len=5120, enable_packet=false, packet_id=13, next_bd=-1, acquire_lock_id=-1, acquire_lock_val=0, release_lock_id=-1, release_lock_val=0, ooo_bd_id=-1, num_dims=2, d0_stride=4, d0_wrap=80, d1_stride=1280, d1_wrap=16 */;
    void* v511 = __runtime_buffer_arg(v81);
    int64_t v512 = (int64_t) v495;
    void* v513 = __runtime_buffer_offset(v511, v512);
    XAie_DmaDesc v514 = __Runtime_dma_bd_config_multidim(v1, v71, v513, 6, 5120, -1, 0, 13, -1, 0, -1, 0, -1, 2, 4, 80, 1280, 16, 0, 0, 0, 0);
    /* Create IO: channel_id=1, bd_id=6, tile=(4,0), direction=S2MM */;
    /* Enable out-of-order BD on tile(4,0) ch=1 dir=S2MM */;
    __Runtime_dma_channel_enable_ooo(v1, v71, 1, DMA_S2MM);
    io v515 = __Runtime_dma_createio_4(v71, v514, 1, 6, DMA_S2MM);
    ioevent v516 = __Runtime_startio(v1, v515, 3, 4);
    /* Wait for 1 event(s) */;
    __Runtime_wait(v1, v380);
    /* Wait for 1 event(s) */;
    __Runtime_wait(v1, v384);
    /* Wait for 1 event(s) */;
    __Runtime_wait(v1, v388);
    /* Wait for 1 event(s) */;
    __Runtime_wait(v1, v392);
    /* Wait for 1 event(s) */;
    __Runtime_wait(v1, v400);
    /* Wait for 1 event(s) */;
    __Runtime_wait(v1, v423);
    /* Wait for 1 event(s) */;
    __Runtime_wait(v1, v431);
    /* Wait for 1 event(s) */;
    __Runtime_wait(v1, v454);
    /* Wait for 1 event(s) */;
    __Runtime_wait(v1, v462);
    /* Wait for 1 event(s) */;
    __Runtime_wait(v1, v485);
    /* Wait for 1 event(s) */;
    __Runtime_wait(v1, v493);
    /* Wait for 1 event(s) */;
    __Runtime_wait(v1, v516);
  }
  /* Wait for 1 event(s) */
  __Runtime_wait(v1, v87);
  /* AieRt debug snapshot (gated by AIE_DEBUG_LOG flag) */
  if (AIEHLC_LOG_ENABLED()) {
    uint8_t _dbg_io_cols[] = {0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 3, 1, 3, 2, 4, 3, 4};
    uint8_t _dbg_io_rows[] = {3, 4, 5, 6, 3, 4, 5, 6, 3, 4, 5, 6, 3, 4, 5, 6, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    uint8_t _dbg_io_chs[] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1};
    uint8_t _dbg_io_bds[] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 4, 4, 4, 4, 2, 2, 2, 2, 4, 4, 4, 4, 2, 2, 2, 2, 4, 4, 4, 4, 2, 2, 2, 2, 4, 4, 4, 4, 0, 0, 0, 0, 1, 2, 1, 7, 1, 1, 11, 6};
    int _dbg_io_dirs[] = {DMA_S2MM, DMA_S2MM, DMA_S2MM, DMA_S2MM, DMA_S2MM, DMA_S2MM, DMA_S2MM, DMA_S2MM, DMA_S2MM, DMA_S2MM, DMA_S2MM, DMA_S2MM, DMA_S2MM, DMA_S2MM, DMA_S2MM, DMA_S2MM, DMA_S2MM, DMA_S2MM, DMA_S2MM, DMA_S2MM, DMA_MM2S, DMA_MM2S, DMA_MM2S, DMA_MM2S, DMA_S2MM, DMA_S2MM, DMA_S2MM, DMA_S2MM, DMA_MM2S, DMA_MM2S, DMA_MM2S, DMA_MM2S, DMA_S2MM, DMA_S2MM, DMA_S2MM, DMA_S2MM, DMA_MM2S, DMA_MM2S, DMA_MM2S, DMA_MM2S, DMA_S2MM, DMA_S2MM, DMA_S2MM, DMA_S2MM, DMA_MM2S, DMA_MM2S, DMA_MM2S, DMA_MM2S, DMA_MM2S, DMA_MM2S, DMA_MM2S, DMA_MM2S, DMA_MM2S, DMA_S2MM, DMA_MM2S, DMA_S2MM, DMA_MM2S, DMA_S2MM, DMA_MM2S, DMA_S2MM};
    uint8_t _dbg_t_cols[] = {0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3};
    uint8_t _dbg_t_rows[] = {3, 4, 5, 6, 3, 4, 5, 6, 3, 4, 5, 6, 3, 4, 5, 6};
    AieRt_DebugSnapshotFromCoords(dev,
        _dbg_io_cols, _dbg_io_rows, _dbg_io_chs, _dbg_io_bds, _dbg_io_dirs, 60,
        _dbg_t_cols, _dbg_t_rows, 16);
  }
  /* Kernel klog only (AIE_KERNEL_CONFIG_TRACE without AIE_DEBUG_LOG) */
  if (AIEHLC_KLOG_ENABLED() && !AIEHLC_LOG_ENABLED()) {
    uint8_t _klog_cols[] = {0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3};
    uint8_t _klog_rows[] = {3, 4, 5, 6, 3, 4, 5, 6, 3, 4, 5, 6, 3, 4, 5, 6};
    __Runtime_read_kernel_logs(dev, _klog_cols, _klog_rows, 16);
  }
  return;
}

__global__ void dskernel_receiver(size_t v1) {
  // the real kernel will be emitted separately

  return;
}



// ===== User source (preserved from original file) =====
#define AIEHLC_TILING_STUBS_DEFINED
namespace aie {
enum class Pattern  { Broadcast = 0, Scatter = 1, Distribute = 1, Multicast = 2, Gather = 3 };
enum class Layout   { Row = 0, Col = 1, Grid = 2 };
enum class Flow     { Default = 0, LeftToRight = 1, RightToLeft = 2 };
enum class PadMaterialize { DDR = 0, Memtile = 1 };
enum class Im2col   { None = 0, Dma = 1 };
enum class LayoutTransform { None = 0, DmaShuffle = 1, CoreShuffle = 2 };
enum class TileMode { Partition = 0, Overlap = 1 };
enum class Objective { MaxArrayUtil = 0, MinLatency = 1, MinDma = 2 };
struct Bytes { int value = 0; };
struct tile_level {
  int tile_size = 0;
  int stride    = 0;
  int rounds    = 0;
};
struct tile_dim {
  int fullsize   = 0;
  int tile_round = 0;
  int tile_size  = 0;
  int stride     = 0;
  int padsize    = 0;
  tile_level slice_tiling;
};
struct SpatialMap {
  Pattern act         = Pattern::Broadcast;
  Pattern wgt         = Pattern::Broadcast;
  Layout  layout      = Layout::Row;
  Flow    merge_order = Flow::Default;
  int     mesh_tiling_group1_dim = -1;
  int     mesh_tiling_group2_dim = -1;
};
struct Materialize {
  PadMaterialize pad    = PadMaterialize::DDR;
  Im2col         im2col = Im2col::None;
};
struct Schedule {
  int   pp_depth  = 2;
  Bytes l1_budget = Bytes{32*1024};
};
struct SpatialPolicy {
  SpatialMap  map;
  Materialize mat;
  Schedule    sched;
};
struct GlobalPolicy {
  int fullconnect_auto = 1;
};
struct GemmSpace {
  SpatialPolicy policy;
  tile_dim m;
  tile_dim n;
  tile_dim k;
  tile_dim d1;
  tile_dim d2;
  tile_dim d3;
  tile_dim d4;
};
struct Conv2dSpace {
  SpatialPolicy policy;
  tile_dim ih;
  tile_dim iw;
  tile_dim ic;
  tile_dim oc;
  tile_dim kh;
  tile_dim kw;
  int stride = 1;
  int pad = 0;
  tile_dim m;
};
struct ConvGeom {
  int in_h=0; int in_w=0; int cin=0; int cout=0;
  int ksize=0; int S=1; int pad_lo=0; int pad_hi=0; int cin_aligned=0;
};
struct Conv2dSpace_Spatial {
  ConvGeom      geom;
  int           out_tile_h = 0;
  int           out_tile_w = 0;
  Objective     objective  = Objective::MaxArrayUtil;
  SpatialPolicy policy;
  tile_dim      d1;
  tile_dim      d2;
};
struct DmaTransform {
  struct Dim { int stride; int wrap; };
  Dim dims[4] = {};
  int num_dims = 0;
  int iter_step = 0;
  int iter_wrap = 0;
  int mode = 0;
  int halo_slice = 0;
  int halo_step = 0;
  int split_dim = 0;
  int raw_h = 0;
  int raw_wc = 0;
  int kernel_h = 0;
  int kernel_w = 0;
  int input_c = 0;
  int stride = 0;
  int ow = 0;
  int oh_per_row = 0;
  static constexpr DmaTransform flat() { return {}; }
  static constexpr DmaTransform im2col(int H, int W, int C,
      int KH, int KW, int S, int P) {
    DmaTransform d;
    int OW = (W + 2*P - KW) / S + 1;
    int OH = (H + 2*P - KH) / S + 1;
    d.dims[0] = {1, KW * C}; d.dims[1] = {W * C, KH}; d.dims[2] = {S * C, OW};
    d.num_dims = 3;
    d.iter_step = W * C * S; d.iter_wrap = OH;
    return d;
  }
  static constexpr DmaTransform dilated_im2col(int H, int W, int C,
      int KH, int KW, int S, int P, int D) {
    DmaTransform d;
    int OW = (W + 2*P - D*(KW-1) - 1) / S + 1;
    int OH = (H + 2*P - D*(KH-1) - 1) / S + 1;
    d.dims[0] = {D * C, KW}; d.dims[1] = {W * C * D, KH}; d.dims[2] = {S * C, OW};
    d.num_dims = 3;
    d.iter_step = W * C * S; d.iter_wrap = OH;
    return d;
  }
  static constexpr DmaTransform pool(int H, int W, int C,
      int KH, int KW, int S, int P) {
    return im2col(H, W, C, KH, KW, S, P);
  }
  static constexpr DmaTransform depthwise_im2col(int H, int W, int C,
      int KH, int KW, int S, int P, int G) {
    DmaTransform d;
    int CPG = C / G;
    int OW = (W + 2*P - KW) / S + 1;
    int OH = (H + 2*P - KH) / S + 1;
    d.dims[0] = {1, KW * CPG}; d.dims[1] = {W * C, KH}; d.dims[2] = {S * C, OW};
    d.num_dims = 3;
    d.iter_step = W * C * S; d.iter_wrap = OH;
    return d;
  }
  static constexpr DmaTransform transpose(int rows, int cols) {
    DmaTransform d;
    d.dims[0] = {cols, rows}; d.dims[1] = {1, cols};
    d.num_dims = 2;
    return d;
  }
  static constexpr DmaTransform chw_to_hwc(int C, int H, int W) {
    DmaTransform d;
    d.dims[0] = {H * W, C};
    d.dims[1] = {1, W};
    d.num_dims = 2;
    d.iter_step = W;
    d.iter_wrap = H;
    return d;
  }
  static constexpr DmaTransform hwc_to_chw(int H, int W, int C) {
    DmaTransform d;
    d.dims[0] = {C, W}; d.dims[1] = {W * C, H};
    d.num_dims = 2;
    d.iter_step = 1; d.iter_wrap = C;
    return d;
  }
  static constexpr DmaTransform spatial(int H, int W, int C,
      int KH, int KW, int S, int P, int R) {
    DmaTransform d;
    int OH = (H + 2*P - KH) / S + 1;
    int oh_per_row = OH / R;
    d.mode = 1;
    d.halo_slice = (oh_per_row - 1) * S + KH;
    d.halo_step  = oh_per_row * S;
    d.split_dim  = 0;
    d.raw_h  = H;
    d.raw_wc = W * C;
    d.kernel_h = KH; d.kernel_w = KW; d.input_c = C; d.stride = S;
    d.ow = (W + 2*P - KW) / S + 1; d.oh_per_row = oh_per_row;
    return d;
  }
};
struct ConvTiling {
  static constexpr DmaTransform spatial(int H, int W, int C,
      int KH, int KW, int S, int P, int R) {
    return DmaTransform::spatial(H, W, C, KH, KW, S, P, R);
  }
};
template<typename T, auto Space, DmaTransform D = DmaTransform::flat()> struct port { using type = T; };
template<typename T> constexpr int get_num_rounds(T) { return 0; }
template<typename T> constexpr int get_buffer_size(T) { return 0; }
constexpr int get_tile_rows() { return 0; }
constexpr int get_tile_cols() { return 0; }
constexpr int get_k_dim() { return 0; }
constexpr int get_tile_m() { return 0; }
constexpr int get_tile_n() { return 0; }
constexpr int get_effective_k() { return 0; }
constexpr int get_k_rounds() { return 0; }
constexpr int get_spatial_m_rounds() { return 0; }
constexpr int get_spatial_n_rounds() { return 0; }
template<typename T> constexpr int get_spatial_multiple_rounds(T) { return 0; }
constexpr int get_kernel_h() { return 0; }
constexpr int get_kernel_w() { return 0; }
constexpr int get_input_c() { return 0; }
constexpr int get_stride() { return 0; }
constexpr int get_ow() { return 0; }
constexpr int get_oh_per_row() { return 0; }
constexpr int get_halo_slice() { return 0; }
}
struct aiePartition {
    int startCol, endCol, startRow, endRow;
};
struct aieMesh {
    int rows, cols;
    aiePartition partition;
    int meshId;
};
void routing(XAie_DevInst* dev) __attribute__((weak));
struct aieArray {
    int nextMeshId = 0;
    XAie_DevInst* _dev = nullptr;
    aieMesh partition(aiePartition p, int rows, int cols) {
        int meshId = nextMeshId++;
        _dev = __Runtime_init_mesh_partition_with(meshId, p.startCol, p.endCol - p.startCol + 1, routing);
        return aieMesh{rows, cols, p, meshId};
    }
    aieMesh partition(int rows, int cols) {
        int meshId = nextMeshId++;
        _dev = __Runtime_init_mesh_partition_with(meshId, 0, cols, routing);
        return aieMesh{rows, cols, {0, cols - 1, 0, rows - 1}, meshId};
    }
    void* alloc(size_t size) { return __Runtime_alloc_buffer(_dev, size); }
    void free(void* ptr) { __Runtime_free_buffer(_dev, ptr); }
    void synchronizecpu(void* ptr, size_t size) { __Runtime_sync_for_cpu(_dev, ptr, size); }
};
struct aieDim {
    int rows, cols;
    aiePartition partition;
    bool hasPartition;
    aieDim(int r, int c) : rows(r), cols(c), partition{-1,-1,-1,-1}, hasPartition(false) {}
    aieDim(int r, int c, aiePartition p) : rows(r), cols(c), partition(p), hasPartition(true) {}
};
inline void aieSetDevice(int) {}
inline void aieDeviceSynchronize() {}
extern unsigned char _binary_kernel_gemm_kacc_start[];
extern unsigned char _binary_kernel_gemm_kacc_end[];
extern unsigned int _binary_kernel_gemm_kacc_size;

inline void __aie_launch(const char* kernel, aieMesh mesh, void* _t0, size_t _s0, void* _t1, size_t _s1, void* _t2, size_t _s2, ...) {
    XAie_DevInst* dev = __Runtime_get_partition_dev(mesh.meshId);
    __Runtime_set_kernel_elf(_binary_kernel_gemm_kacc_start);
    __Runtime_sync_for_dev_deferred(dev, _t0, _s0);
    __Runtime_sync_for_dev_deferred(dev, _t1, _s1);
    __Runtime_sync_for_dev_deferred(dev, _t2, _s2);
    host_canonicalized(dev, _t0, _t1, _t2);
    __Runtime_dsync_drain();
}
inline void __aie_launch(const char* kernel, aieDim mesh, void* _t0, size_t _s0, void* _t1, size_t _s1, void* _t2, size_t _s2, ...) {
    (void)kernel;
    XAie_DevInst* dev;
    if (mesh.hasPartition) {
        dev = __Runtime_explicit_init_partition(mesh.partition.startCol, mesh.partition.endCol - mesh.partition.startCol + 1);
    } else {
        dev = __Runtime_explicit_init();
    }
    __Runtime_set_kernel_elf(_binary_kernel_gemm_kacc_start);
    __Runtime_sync_for_dev_deferred(dev, _t0, _s0);
    __Runtime_sync_for_dev_deferred(dev, _t1, _s1);
    __Runtime_sync_for_dev_deferred(dev, _t2, _s2);
    host_canonicalized(dev, _t0, _t1, _t2);
    __Runtime_dsync_drain();
    __Runtime_explicit_teardown(dev);
}
// Stub type declarations for Clang parsing (function body skipped via #ifdef KERNEL_COMPILE)
#ifndef AIEHLC_STUBS_DEFINED
#define AIEHLC_STUBS_DEFINED
template<typename T> struct input_window {};
template<typename T> struct output_window {};
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
inline void klog(const char*, int) {}
template<typename T> inline void* acquire_input_window(T*) { return (void*)0; }
template<typename T> inline void* acquire_output_window(T*) { return (void*)0; }
template<typename T> inline void release_input_window(T*) {}
template<typename T> inline void release_output_window(T*) {}
#define BUF_SZ 16
#endif

// CUDA-style AIE API stubs for Clang parsing
#ifndef AIEHLC_TILING_STUBS_DEFINED
#define AIEHLC_TILING_STUBS_DEFINED
namespace aie {
enum class Pattern  { Broadcast = 0, Scatter = 1, Distribute = 1, Multicast = 2, Gather = 3 };
enum class Layout   { Row = 0, Col = 1, Grid = 2 };
enum class Flow     { Default = 0, LeftToRight = 1, RightToLeft = 2 };
enum class PadMaterialize { DDR = 0, Memtile = 1 };
enum class Im2col   { None = 0, Dma = 1 };
enum class LayoutTransform { None = 0, DmaShuffle = 1, CoreShuffle = 2 };
enum class TileMode { Partition = 0, Overlap = 1 };
enum class Objective { MaxArrayUtil = 0, MinLatency = 1, MinDma = 2 };
struct Bytes { int value = 0; };
struct tile_level {
  int tile_size = 0;
  int stride    = 0;
  int rounds    = 0;
};
struct tile_dim {
  int fullsize   = 0;
  int tile_round = 0;
  int tile_size  = 0;
  int stride     = 0;
  int padsize    = 0;
  tile_level slice_tiling;
};
struct SpatialMap {
  Pattern act         = Pattern::Broadcast;
  Pattern wgt         = Pattern::Broadcast;
  Layout  layout      = Layout::Row;
  Flow    merge_order = Flow::Default;
  int     mesh_tiling_group1_dim = -1;
  int     mesh_tiling_group2_dim = -1;
};
struct Materialize {
  PadMaterialize pad    = PadMaterialize::DDR;
  Im2col         im2col = Im2col::None;
};
struct Schedule {
  int   pp_depth  = 2;
  Bytes l1_budget = Bytes{32*1024};
};
struct SpatialPolicy {
  SpatialMap  map;
  Materialize mat;
  Schedule    sched;
};
struct GlobalPolicy {
  int fullconnect_auto = 1;
};
struct GemmSpace {
  SpatialPolicy policy;
  tile_dim m;
  tile_dim n;
  tile_dim k;
  tile_dim d1;
  tile_dim d2;
  tile_dim d3;
  tile_dim d4;
};
struct Conv2dSpace {
  SpatialPolicy policy;
  tile_dim ih;
  tile_dim iw;
  tile_dim ic;
  tile_dim oc;
  tile_dim kh;
  tile_dim kw;
  int stride = 1;
  int pad = 0;
  tile_dim m;
};
struct ConvGeom {
  int in_h=0; int in_w=0; int cin=0; int cout=0;
  int ksize=0; int S=1; int pad_lo=0; int pad_hi=0; int cin_aligned=0;
};
struct Conv2dSpace_Spatial {
  ConvGeom      geom;
  int           out_tile_h = 0;
  int           out_tile_w = 0;
  Objective     objective  = Objective::MaxArrayUtil;
  SpatialPolicy policy;
  tile_dim      d1;
  tile_dim      d2;
};
struct DmaTransform {
  struct Dim { int stride; int wrap; };
  Dim dims[4] = {};
  int num_dims = 0;
  int iter_step = 0;
  int iter_wrap = 0;
  int mode = 0;
  int halo_slice = 0;
  int halo_step = 0;
  int split_dim = 0;
  int raw_h = 0;
  int raw_wc = 0;
  int kernel_h = 0;
  int kernel_w = 0;
  int input_c = 0;
  int stride = 0;
  int ow = 0;
  int oh_per_row = 0;
  static constexpr DmaTransform flat() { return {}; }
  static constexpr DmaTransform im2col(int H, int W, int C,
      int KH, int KW, int S, int P) {
    DmaTransform d;
    int OW = (W + 2*P - KW) / S + 1;
    int OH = (H + 2*P - KH) / S + 1;
    d.dims[0] = {1, KW * C}; d.dims[1] = {W * C, KH}; d.dims[2] = {S * C, OW};
    d.num_dims = 3;
    d.iter_step = W * C * S; d.iter_wrap = OH;
    return d;
  }
  static constexpr DmaTransform dilated_im2col(int H, int W, int C,
      int KH, int KW, int S, int P, int D) {
    DmaTransform d;
    int OW = (W + 2*P - D*(KW-1) - 1) / S + 1;
    int OH = (H + 2*P - D*(KH-1) - 1) / S + 1;
    d.dims[0] = {D * C, KW}; d.dims[1] = {W * C * D, KH}; d.dims[2] = {S * C, OW};
    d.num_dims = 3;
    d.iter_step = W * C * S; d.iter_wrap = OH;
    return d;
  }
  static constexpr DmaTransform pool(int H, int W, int C,
      int KH, int KW, int S, int P) {
    return im2col(H, W, C, KH, KW, S, P);
  }
  static constexpr DmaTransform depthwise_im2col(int H, int W, int C,
      int KH, int KW, int S, int P, int G) {
    DmaTransform d;
    int CPG = C / G;
    int OW = (W + 2*P - KW) / S + 1;
    int OH = (H + 2*P - KH) / S + 1;
    d.dims[0] = {1, KW * CPG}; d.dims[1] = {W * C, KH}; d.dims[2] = {S * C, OW};
    d.num_dims = 3;
    d.iter_step = W * C * S; d.iter_wrap = OH;
    return d;
  }
  static constexpr DmaTransform transpose(int rows, int cols) {
    DmaTransform d;
    d.dims[0] = {cols, rows}; d.dims[1] = {1, cols};
    d.num_dims = 2;
    return d;
  }
  static constexpr DmaTransform chw_to_hwc(int C, int H, int W) {
    DmaTransform d;
    d.dims[0] = {H * W, C};
    d.dims[1] = {1, W};
    d.num_dims = 2;
    d.iter_step = W;
    d.iter_wrap = H;
    return d;
  }
  static constexpr DmaTransform hwc_to_chw(int H, int W, int C) {
    DmaTransform d;
    d.dims[0] = {C, W}; d.dims[1] = {W * C, H};
    d.num_dims = 2;
    d.iter_step = 1; d.iter_wrap = C;
    return d;
  }
  static constexpr DmaTransform spatial(int H, int W, int C,
      int KH, int KW, int S, int P, int R) {
    DmaTransform d;
    int OH = (H + 2*P - KH) / S + 1;
    int oh_per_row = OH / R;
    d.mode = 1;
    d.halo_slice = (oh_per_row - 1) * S + KH;
    d.halo_step  = oh_per_row * S;
    d.split_dim  = 0;
    d.raw_h  = H;
    d.raw_wc = W * C;
    d.kernel_h = KH; d.kernel_w = KW; d.input_c = C; d.stride = S;
    d.ow = (W + 2*P - KW) / S + 1; d.oh_per_row = oh_per_row;
    return d;
  }
};
struct ConvTiling {
  static constexpr DmaTransform spatial(int H, int W, int C,
      int KH, int KW, int S, int P, int R) {
    return DmaTransform::spatial(H, W, C, KH, KW, S, P, R);
  }
};
template<typename T, auto Space, DmaTransform D = DmaTransform::flat()> struct port { using type = T; };
template<typename T> constexpr int get_num_rounds(T) { return 0; }
template<typename T> constexpr int get_buffer_size(T) { return 0; }
constexpr int get_tile_rows() { return 0; }
constexpr int get_tile_cols() { return 0; }
constexpr int get_data_row() { return 0; }
constexpr int get_data_col() { return 0; }
constexpr int get_k_dim() { return 0; }
constexpr int get_tile_m() { return 0; }
constexpr int get_tile_n() { return 0; }
constexpr int get_effective_k() { return 0; }
constexpr int get_k_rounds() { return 0; }
constexpr int get_spatial_m_rounds() { return 0; }
constexpr int get_spatial_n_rounds() { return 0; }
template<typename T> constexpr int get_spatial_multiple_rounds(T) { return 0; }
template<typename T> constexpr int get_arg_per_round_size_in_dim(int, T) { return 0; }
template<typename T> constexpr int get_arg_total_rounds_in_dim(int, T) { return 0; }
constexpr int get_kernel_h() { return 0; }
constexpr int get_kernel_w() { return 0; }
constexpr int get_input_c() { return 0; }
constexpr int get_stride() { return 0; }
constexpr int get_ow() { return 0; }
constexpr int get_oh_per_row() { return 0; }
constexpr int get_halo_slice() { return 0; }
}
struct aiePartition {
    int startCol, endCol, startRow, endRow;
};
struct XAie_DevInst;
extern "C" XAie_DevInst *__Runtime_init_mesh_partition(int meshId, int startCol, int numCols);
extern "C" XAie_DevInst *__Runtime_init_mesh_partition_with(int meshId, int startCol, int numCols, void (*route)(XAie_DevInst *));
void routing(XAie_DevInst *dev) __attribute__((weak));
extern "C" XAie_DevInst *__Runtime_get_partition_dev(int meshId);
extern "C" void *__Runtime_alloc_buffer(XAie_DevInst *dev, __SIZE_TYPE__ size_bytes);
extern "C" void __Runtime_free_buffer(XAie_DevInst *dev, void *ptr);
extern "C" void *__Runtime_Alloc(__SIZE_TYPE__ bytes);
extern "C" void __Runtime_sync_for_cpu(XAie_DevInst *dev, void *ptr, __SIZE_TYPE__ size);
extern "C" void __Runtime_teardown_all();
struct aieMesh {
    int rows, cols;
    aiePartition partition;
    int meshId;
};
struct aieArray {
    int nextMeshId = 0;
    XAie_DevInst* _dev = nullptr;
    aieMesh partition(aiePartition p, int rows, int cols) {
        int meshId = nextMeshId++;
        _dev = __Runtime_init_mesh_partition_with(meshId, p.startCol, p.endCol - p.startCol + 1, routing);
        return aieMesh{rows, cols, p, meshId};
    }
    aieMesh partition(int rows, int cols) {
        int meshId = nextMeshId++;
        _dev = __Runtime_init_mesh_partition_with(meshId, 0, cols, routing);
        return aieMesh{rows, cols, {0, cols - 1, 0, rows - 1}, meshId};
    }
    void* alloc(__SIZE_TYPE__ size) { return __Runtime_alloc_buffer(_dev, size); }
    void free(void* ptr) { __Runtime_free_buffer(_dev, ptr); }
    void synchronizecpu(void* ptr, __SIZE_TYPE__ size) { __Runtime_sync_for_cpu(_dev, ptr, size); }
};
struct aieDim {
    int rows, cols;
    aiePartition partition;
    bool hasPartition;
    aieDim(int r, int c) : rows(r), cols(c), partition{-1,-1,-1,-1}, hasPartition(false) {}
    aieDim(int r, int c, aiePartition p) : rows(r), cols(c), partition(p), hasPartition(true) {}
};
inline void aieSetDevice(int) {}
inline void aieDeviceSynchronize() {}
extern void host_canonicalized(...);
template<typename... Args>
inline void __aie_launch(const char* kernel, aieMesh mesh, Args... args) {
    (void)kernel; (void)mesh; (void)sizeof...(args);
}
template<typename... Args>
inline void __aie_launch(const char* kernel, aieDim mesh, Args... args) {
    (void)kernel; (void)mesh; (void)sizeof...(args);
}
#endif

#define KACC_REPS 8
// kacc_gemm.cc -- generated by kacc_gemm.py: gemm_kacc as a dense GEMM (PDK probe)
#define AIEDNN_KACC_KERNEL 1
#include "kacc_data.h"

// gemm_kacc: K-in-core bf16 GEMM (fp32 accumulator in the c window) on the 4x4 mesh, one C++ template kernel
// instantiated by aiehlc per launch gemm_kacc<CH, KDIM, PX, KW, AROWS, BROWS, TKW, KCH, NKC, NT, S, TR>.
// The template arguments come from the walk geometry (aiegraph kacc_geom -> aiednn.gemm_kacc_targs):
//   CH / KDIM / PX  the instance's channels, K (bf16) and pixels; KW = padded K in int32 words
//   AROWS / BROWS   rows of the a / b ports (CH / PX, swapped when transposed)
//   TKW / KCH / NKC K chunk in words / in bf16 / chunks per tile; NT a-port 16-row tiles, S b sub-tiles per core
//   TR              transposed: a = X, b = W (the mmul's left operand stays W)
                                                                                                                    
// ==== aiehlc trace -> routing: gemm_kacc's ports routed from its traced load_buffer / store_buffer blocks ====
//  routing plan on the 4x4 mesh:
//    a_port (IFM, row broadcast, 16x20 blocks/core): d1[1024]=rows/4@256+i16x16@16 d2[160]=j20x8@8
//    b_port (WEIGHTS, col broadcast, 16x20 blocks/core): d1[320]=cols/4@80 d2[160]=j20x8@8
//    c_port (OFM, gather, 1x16 blocks/core): d1[1024]=rows/4@256+j16x16@16 d2[320]=cols/4@80
constexpr aie::GemmSpace __aie_gemm_kacc_IFM = {
    .policy = {.map = {.act = aie::Pattern::Broadcast, .layout = aie::Layout::Row},
               .mat = {.pad = aie::PadMaterialize::DDR, .im2col = aie::Im2col::None},
               .sched = {.pp_depth = 2, .l1_budget = aie::Bytes{5120}}},
    .d1 = {.fullsize = 1024, .tile_round = 64, .tile_size = 16, .stride = 16},
    .d2 = {.fullsize = 160, .tile_round = 20, .tile_size = 8, .stride = 8}};
constexpr aie::GemmSpace __aie_gemm_kacc_WEIGHTS = {
    .policy = {.map = {.wgt = aie::Pattern::Broadcast, .layout = aie::Layout::Col},
               .mat = {.pad = aie::PadMaterialize::DDR, .im2col = aie::Im2col::None},
               .sched = {.pp_depth = 2, .l1_budget = aie::Bytes{5120}}},
    .d1 = {.fullsize = 320, .tile_round = 4, .tile_size = 80, .stride = 80},
    .d2 = {.fullsize = 160, .tile_round = 20, .tile_size = 8, .stride = 8}};
constexpr aie::GemmSpace __aie_gemm_kacc_OFM = {
    .policy = {.map = {.layout = aie::Layout::Row, .merge_order = aie::Flow::LeftToRight, .mesh_tiling_group1_dim = 1, .mesh_tiling_group2_dim = 2},
               .mat = {.pad = aie::PadMaterialize::DDR, .im2col = aie::Im2col::None},
               .sched = {.pp_depth = 2, .l1_budget = aie::Bytes{5120}}},
    .d1 = {.fullsize = 1024, .tile_round = 64, .tile_size = 16, .stride = 16},
    .d2 = {.fullsize = 320, .tile_round = 4, .tile_size = 80}};
constexpr aie::GlobalPolicy __aie_gemm_kacc_policy = {.fullconnect_auto = 1};
// ==== end aiehlc trace -> routing ====

// Global variables for kernel: gemm_kacc
extern unsigned char _binary_kernel_gemm_kacc_start[];
extern unsigned char _binary_kernel_gemm_kacc_end[];
extern unsigned int _binary_kernel_gemm_kacc_size;



/* script/perf/p0_kacc_host.cc -- cost-model P0 probe host for one gemm_kacc instance (kacc_gemm.py, KACC_HOST_FILE).
 * Same data/numerics as src/aiednn/gemm/kacc_bench_gemm_host.cc, plus:
 *   - "BOARD machine=aarch64 midr=0x%08lx" (MIDR_EL1) first; the wrapper appends " board=<name>";
 *   - AIE core perf counters armed at every launch (--profiling build + g_runtime_debug_level 0x100) and read once per
 *     timed launch AFTER the launch returned (wait_io done, Y invalidated), outside every timed region:
 *     "[PERF] tag=<launch> ..." per core tile + "[PERFSUM] tag=<launch> ...";
 *   - per launch the host regions of kacc_chunk_board.cc uc_launch (flush W/X/Y | launch | invalidate Y, pmccntr),
 *     and the runtime phase / wait_io accumulators (delta over the timed pass) printed in the
 *     src/aiegraph/runtime/main_launch_line.c format ("LAUNCH split" / "LAUNCH split_detail calls").
 * KACC_REPS (default 8) timed passes of the KACC_S launches after one warm-up pass. */
#ifndef KACC_REPS
#define KACC_REPS 8
#endif
void __Runtime_ctrl_high_throughput_enable(int on); /* C++ linkage, as aie_runtime.h:936 */
void __Runtime_phase_cycles(unsigned long long *cyc, unsigned int *calls);
void __Runtime_wait_io_cycles(unsigned long long *cycles, unsigned int *calls);
void __Runtime_core_perf_report(int tag);
void __Runtime_launch_words(unsigned long long *mmio_w, unsigned long long *cp_w, unsigned long long *ph_w,
                            unsigned int *bd_n, unsigned int *rst_n, unsigned long long *miss);
int g_runtime_debug_level = 0x100; /* AIE_DEBUG_FLAG_CORE_PERF_COUNTER (aie_runtime.h:69); overrides the weak 0 */

static inline unsigned long long p0_pmc(void) {
    unsigned long long v;
#if defined(__aarch64__)
    __asm__ volatile("isb; mrs %0, pmccntr_el0" : "=r"(v));
#else
    v = 0ULL;
#endif
    return v;
}
static inline unsigned long long p0_rt_cyc(void) {
    unsigned long long ph[4] = {0, 0, 0, 0}, w = 0;
    unsigned int pn[4] = {0, 0, 0, 0}, wn = 0;
    __Runtime_phase_cycles(ph, pn);
    __Runtime_wait_io_cycles(&w, &wn);
    return ph[0] + ph[1] + ph[2] + ph[3] + w;
}

/* device index of Y[m][n] in launch s's fp32 c buffer (unet_aie uc_conv un-tiling, P4bx [s][mi][ni][8x8]) */
static size_t kacc_yidx(int m, int n) {
    const int s = KACC_SPLIT_PX ? m / KACC_PXS : n / KACC_CHS;
    const int ml = KACC_SPLIT_PX ? m % KACC_PXS : m, nl = KACC_SPLIT_PX ? n : n % KACC_CHS;
    const int Ml = KACC_PXS, Nl = KACC_CHS, r = nl & 15, c = ml & 15;
    size_t base = (size_t)s * Ml * Nl, i;
    if (!KACC_TR) {
        const int q = Ml / 4, cc = ml % q;
        const int wi = (cc >> 4) * 256 + ((r >> 3) * 2 + (c >> 3)) * 64 + (r & 7) * 8 + (c & 7);
        i = (size_t)((nl & ~15) + wi / q) * Ml + (ml - cc) + wi % q;
    } else {
        const int q = Nl / 4, cc = nl % q;
        const int wi = (cc >> 4) * 256 + ((r >> 3) * 2 + (c >> 3)) * 64 + (r & 7) * 8 + (c & 7);
        i = (size_t)((ml & ~15) + wi / q) * Nl + (nl - cc) + wi % q;
    }
    return base + i;
}

int main() {
#if defined(__aarch64__)
    {
        unsigned long midr;
        __asm__ volatile("mrs %0, midr_el1" : "=r"(midr));
        printf("BOARD machine=aarch64 midr=0x%08lx\n", midr);
    }
#else
    printf("BOARD machine=other midr=0x00000000\n");
#endif
    printf("=== P0 gemm_kacc M=%d (pad %d) K=%d N=%d: %d launch(es) of gemm_kacc<320, 320, 1024, 160, 1024, 320, 8, 16, 20, 16, 5, 1> per pass x %d timed passes, 4x4 mesh, "
           "core perf counters on ===\n",
           KACC_M, KACC_MP, KACC_K, KACC_N, KACC_S, KACC_REPS);
    __Runtime_ctrl_high_throughput_enable(1);
    aieSetDevice(0);
    aieArray device;
    aieMesh mesh = device.partition({2, 6, 0, 6}, 4, 4);
    const size_t wbytes = (size_t)KACC_N * KACC_KP * 2, xbytes = (size_t)KACC_MP * KACC_KP * 2;
    const size_t ybytes = (size_t)KACC_N * KACC_MP * 4;
    int32_t *Wd = (int32_t *)__Runtime_Alloc(wbytes);
    int32_t *Xd = (int32_t *)__Runtime_Alloc(xbytes);
    int32_t *Yd = (int32_t *)__Runtime_Alloc(ybytes);
    if (!Wd || !Xd || !Yd) {
        printf("KCMP %s data=%s y_sha256=%s cos=0.000000 FAIL (alloc)\n", KACC_TAG, KACC_DATA, KACC_Y_SHA);
        return 1;
    }
    memcpy(Wd, kacc_w_bin, wbytes);
    memcpy(Xd, kacc_x_bin, xbytes);
    Xil_DCacheFlushRange((UINTPTR)Wd, wbytes);
    Xil_DCacheFlushRange((UINTPTR)Xd, xbytes);
    /* warm-up pass: kernel load + first-launch config, not counted */
    memset(Yd, 0xff, ybytes);
    Xil_DCacheFlushRange((UINTPTR)Yd, ybytes);
    for (int s = 0; s < KACC_S; s++) {
        int32_t *Ws = Wd + (KACC_SPLIT_PX ? 0 : (size_t)s * KACC_CHS * KACC_KP / 2);
        int32_t *Xs = Xd + (KACC_SPLIT_PX ? (size_t)s * KACC_PXS * KACC_KP / 2 : 0);
        int32_t *Ys = Yd + (size_t)s * KACC_CHS * KACC_PXS;
        __aie_launch("gemm_kacc", mesh, Xs, (size_t)655360, Ws, (size_t)204800, Ys, (size_t)1310720);
    }
    Xil_DCacheInvalidateRange((UINTPTR)Yd, ybytes);
    printf("[P0] warmup done launches=%d\n", KACC_S);

    unsigned long long ph0[4] = {0, 0, 0, 0}, ph1[4] = {0, 0, 0, 0}, wio0 = 0, wio1 = 0;
    unsigned int phn0[4] = {0, 0, 0, 0}, phn1[4] = {0, 0, 0, 0}, wion0 = 0, wion1 = 0;
    unsigned long long sp_pre = 0, sp_call = 0, sp_glue = 0, sp_post = 0, ticks = 0;
    __Runtime_phase_cycles(ph0, phn0);
    __Runtime_wait_io_cycles(&wio0, &wion0);
    unsigned long long wm0 = 0, wc0 = 0, wp0[4] = {0, 0, 0, 0}, wx0 = 0;
    unsigned int wbd0 = 0, wrs0 = 0;
    __Runtime_launch_words(&wm0, &wc0, wp0, &wbd0, &wrs0, &wx0);
    XTime g0, g1;
    XTime_GetTime(&g0);
    const unsigned long long pmc0 = p0_pmc();
    int launches = 0;
    for (int rep = 0; rep < KACC_REPS; rep++) {
        memset(Yd, 0xff, ybytes); /* sentinel: an element the pass does not write is nonfinite */
        for (int s = 0; s < KACC_S; s++) {
            int32_t *Ws = Wd + (KACC_SPLIT_PX ? 0 : (size_t)s * KACC_CHS * KACC_KP / 2);
            int32_t *Xs = Xd + (KACC_SPLIT_PX ? (size_t)s * KACC_PXS * KACC_KP / 2 : 0);
            int32_t *Ys = Yd + (size_t)s * KACC_CHS * KACC_PXS;
            const size_t yb = (size_t)KACC_CHS * KACC_PXS * 4;
            XTime lt0, lt1;
            XTime_GetTime(&lt0);
            const unsigned long long a0 = p0_pmc();
            Xil_DCacheFlushRange((UINTPTR)Ws, (size_t)KACC_CHS * KACC_KP * 2);
            Xil_DCacheFlushRange((UINTPTR)Xs, (size_t)KACC_PXS * KACC_KP * 2);
            Xil_DCacheFlushRange((UINTPTR)Ys, yb);
            const unsigned long long r0 = p0_rt_cyc(), a1 = p0_pmc();
            __aie_launch("gemm_kacc", mesh, Xs, (size_t)655360, Ws, (size_t)204800, Ys, (size_t)1310720);
            const unsigned long long a2 = p0_pmc(), r1 = p0_rt_cyc();
            Xil_DCacheInvalidateRange((UINTPTR)Ys, yb);
            const unsigned long long a3 = p0_pmc();
            XTime_GetTime(&lt1);
            sp_pre += a1 - a0;
            sp_call += a2 - a1;
            sp_glue += (a2 - a1) > (r1 - r0) ? (a2 - a1) - (r1 - r0) : 0ULL;
            sp_post += a3 - a2;
            ticks += (unsigned long long)(lt1 - lt0);
            /* outside the timed regions: the counters armed at this launch (core done: wait_io returned) */
            __Runtime_core_perf_report(launches);
            launches++;
        }
    }
    const unsigned long long pmc1 = p0_pmc();
    XTime_GetTime(&g1);
    __Runtime_phase_cycles(ph1, phn1);
    __Runtime_wait_io_cycles(&wio1, &wion1);
    unsigned long long wm1 = 0, wc1 = 0, wp1[4] = {0, 0, 0, 0}, wx1 = 0;
    unsigned int wbd1 = 0, wrs1 = 0;
    __Runtime_launch_words(&wm1, &wc1, wp1, &wbd1, &wrs1, &wx1);
    unsigned long long ph[4];
    unsigned int phn[4];
    for (int i = 0; i < 4; i++) {
        ph[i] = ph1[i] - ph0[i];
        phn[i] = phn1[i] - phn0[i];
    }
    const unsigned long long wio = wio1 - wio0;
    const unsigned int wion = wion1 - wion0;
    const double g_s = (double)(g1 - g0) / (double)COUNTS_PER_SECOND;
    const unsigned long long pmu_hz = g_s > 0.0 ? (unsigned long long)((double)(pmc1 - pmc0) / g_s + 0.5) : 0ULL;
    const double cms = pmu_hz ? 1000.0 / (double)pmu_hz : 0.0;
    const double nl = launches ? (double)launches : 1.0;
    const double aie_ms = 1000.0 * (double)ticks / (double)COUNTS_PER_SECOND;
    const unsigned long long c_kl = ph[0], c_bd = ph[1] + sp_glue, c_si = ph[3] + sp_pre, c_ce = ph[2],
                             c_wi = wio + sp_post;
    const double rt_ms = (double)(ph[0] + ph[1] + ph[2] + ph[3] + wio) * cms;
    const double sum_ms = (double)(c_kl + c_bd + c_si + c_ce + c_wi) * cms;
    printf("[BENCH] aie_wall_ms=%.3f launches=%d gmac=%.4f\n", aie_ms, launches, KACC_GMAC * KACC_REPS);
    printf("LAUNCH split kload=%.3f bdcfg=%.3f startio=%.3f coreen=%.3f wait_io=%.3f core_reset=%.3f "
           "ms_per_launch=%.3f pmu_hz=%llu sum_vs_aie_wall=%.3f\n",
           c_kl * cms / nl, c_bd * cms / nl, c_si * cms / nl, c_ce * cms / nl, c_wi * cms / nl, 0.0, aie_ms / nl,
           pmu_hz, aie_ms > 0.0 ? sum_ms / aie_ms : 0.0);
    printf("LAUNCH split_detail calls kload=%u bdcfg=%u startio=%u coreen=%u wait_io=%u core_reset=0 "
           "rt_bdcfg=%.4f rt_startio=%.4f rt_wait_io=%.4f "
           "host_glue_bdcfg=%.4f host_flush_startio=%.4f host_inval_wait_io=%.4f gemm_call=%.4f "
           "rt_phase_sum_ms=%.3f phase_sum_ms=%.3f aie_wall_ms=%.3f unattributed_ms_per_launch=%.4f "
           "pmc_delta=%llu xtime_s=%.3f launches=%d\n",
           phn[0], phn[1], phn[3], phn[2], wion, ph[1] * cms / nl, ph[3] * cms / nl, wio * cms / nl,
           sp_glue * cms / nl, sp_pre * cms / nl, sp_post * cms / nl, sp_call * cms / nl, rt_ms, sum_ms, aie_ms,
           (aie_ms - sum_ms) / nl, pmc1 - pmc0, g_s, launches);
#if defined(AIEHLC_PROFILING) && AIEHLC_PROFILING
    /* M1.1 launch words per launch over the timed pass (main_launch_line.c format; PH_KLOAD=0 PH_BDCFG=1
     * PH_COREEN=2 PH_STARTIO=3) */
    printf("LAUNCH split_words mmio_w=%.1f cp_w=%.1f bd_n=%.1f rst_n=%.1f kload_w=%.1f bdcfg_w=%.1f startio_w=%.1f coreen_w=%.1f\n",
           (double)(wm1 - wm0) / nl, (double)(wc1 - wc0) / nl, (double)(wbd1 - wbd0) / nl, (double)(wrs1 - wrs0) / nl,
           (double)(wp1[0] - wp0[0]) / nl, (double)(wp1[1] - wp0[1]) / nl, (double)(wp1[3] - wp0[3]) / nl,
           (double)(wp1[2] - wp0[2]) / nl);
    printf("[P0] split_words_miss=%llu\n", wx1 - wx0);
#endif
    const float *ref = (const float *)kacc_y_bin; /* [M][N] */
    double dot = 0, na = 0, nr = 0, maxabs = 0;
    int nonfinite = 0;
    for (int m = 0; m < KACC_M; m++)
        for (int n = 0; n < KACC_N; n++) {
            float f;
            memcpy(&f, &Yd[kacc_yidx(m, n)], 4);
            const double o = (double)f, r = ref[(size_t)m * KACC_N + n];
            if (!(o == o) || o > 1e30 || o < -1e30) {
                nonfinite++;
                continue;
            }
            dot += o * r;
            na += o * o;
            nr += r * r;
            if (fabs(o - r) > maxabs)
                maxabs = fabs(o - r);
        }
    const double cs = nonfinite ? 0.0 : dot / (sqrt(na) * sqrt(nr) + 1e-30);
    const int pass = nonfinite == 0 && cs >= 0.9995;
    printf("[PDK] maxabs_vs_ref=%.6g nonfinite=%d M=%d N=%d K=%d\n", maxabs, nonfinite, KACC_M, KACC_N, KACC_K);
    printf("KCMP %s data=%s y_sha256=%s cos=%.6f %s\n", KACC_TAG, KACC_DATA, KACC_Y_SHA, cs, pass ? "PASS" : "FAIL");
    fflush(stdout);
    return pass ? 0 : 1;
}

