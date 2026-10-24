#include "aie_runtime_xaie_seq.h"

#include <stdio.h>

AieRC rt_seq_bd_desc(XAie_DevInst *dev, XAie_LocType tile, uint64_t dma_addr, const rt_seq_bd_args *a,
                     XAie_DmaDesc *desc) {
    AieRC rc = XAie_DmaDescInit(dev, desc, tile);
    if (rc != XAIE_OK)
        return rc;
    int max_addr_dims = 3;
    int num_dims = a->num_dims > (a->ooo ? 3 : 4) ? (a->ooo ? 3 : 4) : a->num_dims;
    int addr_dims = num_dims <= max_addr_dims ? num_dims : max_addr_dims;
    XAie_DmaDimDesc dims[3];
    for (int i = 0; i < addr_dims; i++) {
        /* IR strides are in byte units; XAie expects 32-bit word units. */
        if (a->stride[i] % 4 != 0)
            printf("[aie_runtime] ERROR: dim_stride[%d]=%d not divisible by 4 (must be 32-bit aligned)\n", i,
                   a->stride[i]);
        dims[i].AieMlDimDesc.StepSize = (uint32_t)(a->stride[i] / 4);
        dims[i].AieMlDimDesc.Wrap = (uint16_t)a->wrap[i];
    }
    XAie_DmaTensor tensor;
    tensor.NumDim = (uint8_t)addr_dims;
    tensor.Dim = dims;
    XAie_DmaSetMultiDimAddr(desc, &tensor, dma_addr, (uint32_t)a->len);
    if (a->ooo) {
        if (a->iter_wrap > 1)
            XAie_DmaSetBdIteration(desc, a->iter_step_size / 4, a->iter_wrap, 0);
    } else if (num_dims == 4) {
        XAie_DmaSetBdIteration(desc, a->stride[3] / 4, a->wrap[3], 0);
    }
    if (a->acquire_lock_id >= 0 && a->release_lock_id >= 0)
        XAie_DmaSetLock(desc, XAie_LockInit(a->acquire_lock_id, a->acquire_lock_val),
                        XAie_LockInit(a->release_lock_id, a->release_lock_val));
    if (a->next_bd >= 0)
        XAie_DmaSetNextBd(desc, (uint8_t)a->next_bd, XAIE_ENABLE);
    if (a->enable_packet)
        XAie_DmaSetPkt(desc, XAie_PacketInit(a->packet_id, 0));
    if (a->out_of_order_bd_id >= 0)
        XAie_DmaSetOutofOrderBdId(desc, (uint8_t)a->out_of_order_bd_id);
    /* Shim BDs move DDR data over the NoC: request long AXI bursts. Left at the
     * zero-initialized default the BD issues 4-beat (64-byte) transactions; on
     * conv2dstem's output gather (4 cores per shim S2MM channel) that made the
     * shim drain the bottleneck the cores stalled on. Only the shim DMA has AXI
     * fields -- core-tile BDs keep none (XAie_DmaSetAxi would reject them). The
     * offline AOT tool links this same function, so precomputed shim-BD packets
     * carry the same words. */
    {
        uint8_t tt = dev->DevOps->GetTTypefromLoc(dev, tile);
        if (tt == XAIEGBL_TILE_TYPE_SHIMNOC)
            (void)XAie_DmaSetAxi(desc, 0U, RT_SEQ_SHIM_AXI_BURST, RT_SEQ_SHIM_AXI_QOS, RT_SEQ_SHIM_AXI_CACHE, 0U);
    }
    XAie_DmaEnableBd(desc);
    return XAIE_OK;
}

AieRC rt_seq_channel_ooo_desc(XAie_DevInst *dev, XAie_LocType tile, XAie_DmaChannelDesc *desc) {
    AieRC rc = XAie_DmaChannelDescInit(dev, desc, tile);
    if (rc == XAIE_OK)
        rc = XAie_DmaChannelEnOutofOrder(desc, XAIE_ENABLE);
    return rc;
}

AieRC rt_seq_start_queue(XAie_DevInst *dev, XAie_LocType tile, uint8_t channel, XAie_DmaDirection dir, uint8_t bd_id,
                         int32_t repeat) {
    return XAie_DmaChannelSetStartQueue(dev, tile, channel, dir, bd_id, (uint32_t)repeat, XAIE_DISABLE);
}

AieRC rt_seq_shim_bd_addr_fields(XAie_DevInst *dev, XAie_LocType tile, uint64_t addr, uint32_t f[3]) {
    uint8_t tt = dev->DevOps->GetTTypefromLoc(dev, tile);
    if (tt >= XAIEGBL_TILE_TYPE_MAX || !dev->DevProp.DevMod[tt].DmaMod)
        return XAIE_INVALID_TILE;
    const XAie_ShimDmaBuffer *b = &dev->DevProp.DevMod[tt].DmaMod->BdProp->Buffer->ShimDmaBuff;
    f[0] = XAie_SetField(addr >> b->AddrLow.Lsb, b->AddrLow.Lsb, b->AddrLow.Mask);
    f[1] = XAie_SetField(addr >> 32U, b->AddrHigh.Lsb, b->AddrHigh.Mask);
    f[2] = XAie_SetField(addr >> b->AddrExtHigh.Lsb, b->AddrExtHigh.Lsb, b->AddrExtHigh.Mask);
    return XAIE_OK;
}

StrmSwPortType rt_seq_acr_port(acr_port p) {
    switch (p) {
    case ACR_WEST:
        return WEST;
    case ACR_EAST:
        return EAST;
    case ACR_NORTH:
        return NORTH;
    case ACR_SOUTH:
        return SOUTH;
    case ACR_CTRL:
    default:
        return CTRL;
    }
}

uint8_t rt_seq_acr_chan(acr_port p, int is_master, uint8_t idx) {
    if (p == ACR_NORTH)
        return (uint8_t)(is_master ? RT_RES_VFWD_PORT : RT_RES_VRET_PORT);
    if (p == ACR_SOUTH)
        return (uint8_t)(is_master ? RT_RES_VRET_PORT : RT_RES_VFWD_PORT);
    return idx;
}

AieRC rt_seq_row_op(XAie_DevInst *dev, const acr_op *op) {
    XAie_LocType loc = XAie_TileLoc(op->col, op->row);
    uint8_t sidx = rt_seq_acr_chan(op->sport, 0, op->sidx);
    uint8_t midx = rt_seq_acr_chan(op->mport, 1, op->midx);
    switch (op->kind) {
    case ACR_OP_SLOT:
        return XAie_StrmPktSwSlaveSlotEnable(dev, loc, rt_seq_acr_port(op->sport), sidx, op->slot,
                                             XAie_PacketInit(op->pkt_id, 0U), op->mask, op->msel, op->arbiter);
    case ACR_OP_SLAVE_EN:
        return XAie_StrmPktSwSlavePortEnable(dev, loc, rt_seq_acr_port(op->sport), sidx);
    case ACR_OP_MASTER_EN:
        return XAie_StrmPktSwMstrPortEnable(dev, loc, rt_seq_acr_port(op->mport), midx,
                                            op->keep_header ? XAIE_SS_PKT_DONOT_DROP_HEADER : XAIE_SS_PKT_DROP_HEADER,
                                            op->arbiter, op->mselen);
    case ACR_OP_CCT:
        return XAie_StrmConnCctEnable(dev, loc, rt_seq_acr_port(op->sport), sidx, rt_seq_acr_port(op->mport), midx);
    default:
        return XAIE_INVALID_ARGS;
    }
}
