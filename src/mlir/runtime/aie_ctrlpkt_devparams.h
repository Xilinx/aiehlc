/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/
#ifndef AIE_CTRLPKT_DEVPARAMS_H
#define AIE_CTRLPKT_DEVPARAMS_H

#include "aie_ctrlpkt_encode.h"

static inline int rt_cpe_dev_for_gen(int gen, rt_cpe_dev *out) {
    if (gen == 5) {
        out->prog_mem_size = 16U * 1024U;
        out->prog_mem_host_off = 0x00020000U;
        out->data_mem_addr = 0x40000U;
        out->data_mem_size = 64U * 1024U;
        out->core_ctrl_off = 0x00038000U;
        out->core_rst_mask = 0x00000002U;
        out->checkerboard = 0U;
        out->dma_ch_ctrl_off = 0x0001DE00U;
        out->dma_ch_stride = 0x8U;
        out->dma_nch = 2U;
        out->dma_ch_rst_mask = 0x00000002U;
        return 0;
    }
    return -1;
}

#endif
