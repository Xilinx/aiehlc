#include "aie_runtime_resource.h"

/* Static reservation table describing the control fabric's stream-switch
 * footprint. Gen is a parameter: Gen1/2/5 share the same stream-switch slot
 * semantics today (the table is identical), but the rt_res_gen enum + the gen
 * switch in rt_res_table() make a future per-gen divergence a one-line change.
 *
 * Forward fabric (arbiter RT_RES_ARB_CTRL):
 *   ingress-slave slots on WEST and SOUTH (the two head/interior ingress ports):
 *     CONSUME   (pkt=rowidx 0..RT_RES_MAX_ROW_IDX, mask 0x17, msel0)
 *     BCAST     (pkt 0x10, mask 0x10, msel1)
 *     TRANSIT_N (pkt 0x00, mask 0x10, msel2)   [spine head only]
 *   masters: CTRL (mselen 0x3), EAST (0xB), NORTH (0x6)
 * Return fabric (arbiter RT_RES_ARB_RET), each slot on its own slave port:
 *     RET_LOCAL   on CTRL  slave (accept-any pkt=0/mask=0, msel0)
 *     RET_TRANSIT on EAST  slave (msel1)
 *     RET_NORTH   on NORTH slave (msel2)
 *   masters: WEST, SOUTH */

#define RT_MSELEN_CTRL ((1u << RT_RES_MSEL_CONSUME) | (1u << RT_RES_MSEL_BCAST))             /* 0x3 */
#define RT_MSELEN_EAST ((1u << RT_RES_MSEL_CONSUME) | (1u << RT_RES_MSEL_BCAST) | (1u << 3)) /* 0xB */
#define RT_MSELEN_NORTH ((1u << RT_RES_MSEL_BCAST) | (1u << RT_RES_MSEL_TRANSIT_N))          /* 0x6 */
#define RT_MSELEN_RET_HEAD                                                                                             \
    ((1u << RT_RES_MSEL_RET_LOCAL) | (1u << RT_RES_MSEL_RET_TRANSIT) | (1u << RT_RES_MSEL_RET_NORTH)) /* 0x7 */

/* Build the reservation table for @gen into @tab (cap @cap), returns count. All
 * generations share one table today; the switch documents where they diverge. */
static int rt_res_table(rt_res_gen gen, rt_res_entry *tab, int cap) {
    int n = 0;
    (void)gen; /* Gen1/2/5 identical today; switch here to diverge per gen. */

#define RT_ADD(P, M, SLOT, ARB, MSEL, PKT, MASK, RET)                                                                  \
    do {                                                                                                               \
        if (n >= cap)                                                                                                  \
            return n;                                                                                                  \
        tab[n].port = (uint8_t)(P);                                                                                    \
        tab[n].is_master = (uint8_t)(M);                                                                               \
        tab[n].slot = (uint8_t)(SLOT);                                                                                 \
        tab[n].arbiter = (uint8_t)(ARB);                                                                               \
        tab[n].msel = (uint8_t)(MSEL);                                                                                 \
        tab[n].pkt_id = (uint8_t)(PKT);                                                                                \
        tab[n].mask = (uint8_t)(MASK);                                                                                 \
        tab[n].is_ret = (uint8_t)(RET);                                                                                \
        n++;                                                                                                           \
    } while (0)

    /* --- Forward fabric: ingress-slave slots on WEST and SOUTH --- */
    for (int pi = 0; pi < 2; pi++) {
        uint8_t sp = pi == 0 ? RT_RES_PORT_WEST : RT_RES_PORT_SOUTH;
        /* CONSUME enumerates a slot per target-row add-order index. */
        for (uint8_t rowidx = 0; rowidx <= RT_RES_MAX_ROW_IDX; rowidx++)
            RT_ADD(sp, 0, RT_RES_SLOT_CONSUME, RT_RES_ARB_CTRL, RT_RES_MSEL_CONSUME, rowidx, RT_RES_MASK_CONSUME, 0);
        RT_ADD(sp, 0, RT_RES_SLOT_BCAST, RT_RES_ARB_CTRL, RT_RES_MSEL_BCAST, RT_RES_ID_BCAST, RT_RES_MASK_BCAST, 0);
        RT_ADD(sp, 0, RT_RES_SLOT_TRANSIT_N, RT_RES_ARB_CTRL, RT_RES_MSEL_TRANSIT_N, RT_RES_ID_TRANSIT,
               RT_RES_MASK_CLASS, 0);
    }
    /* Forward masters. */
    RT_ADD(RT_RES_PORT_CTRL, 1, RT_RES_NA, RT_RES_ARB_CTRL, RT_MSELEN_CTRL, RT_RES_NA, RT_RES_NA, 0);
    RT_ADD(RT_RES_PORT_EAST, 1, RT_RES_NA, RT_RES_ARB_CTRL, RT_MSELEN_EAST, RT_RES_NA, RT_RES_NA, 0);
    RT_ADD(RT_RES_PORT_NORTH, 1, RT_RES_NA, RT_RES_ARB_CTRL, RT_MSELEN_NORTH, RT_RES_NA, RT_RES_NA, 0);

    /* --- Return fabric: one accept-any slot per return slave port --- */
    RT_ADD(RT_RES_PORT_CTRL, 0, RT_RES_SLOT_RET_LOCAL, RT_RES_ARB_RET, RT_RES_MSEL_RET_LOCAL, 0, 0, 1);
    RT_ADD(RT_RES_PORT_EAST, 0, RT_RES_SLOT_RET_TRANSIT, RT_RES_ARB_RET, RT_RES_MSEL_RET_TRANSIT, 0, 0, 1);
    RT_ADD(RT_RES_PORT_NORTH, 0, RT_RES_SLOT_RET_NORTH, RT_RES_ARB_RET, RT_RES_MSEL_RET_NORTH, 0, 0, 1);
    /* Return masters (mselen supersets; individual heads pull subsets). */
    RT_ADD(RT_RES_PORT_WEST, 1, RT_RES_NA, RT_RES_ARB_RET, RT_MSELEN_RET_HEAD, RT_RES_NA, RT_RES_NA, 1);
    RT_ADD(RT_RES_PORT_SOUTH, 1, RT_RES_NA, RT_RES_ARB_RET, RT_MSELEN_RET_HEAD, RT_RES_NA, RT_RES_NA, 1);

#undef RT_ADD
    return n;
}

/* Full table size upper bound: 2 ingress ports * (4 CONSUME + BCAST + TRANSIT) =
 * 12, + 3 fwd masters + 3 ret slots + 2 ret masters = 20. */
#define RT_RES_TABLE_MAX 32

int __Runtime_res_port_reserved(rt_res_gen gen, int col, int row, uint8_t port, uint8_t is_master, rt_res_entry *out,
                                int cap) {
    (void)col;
    (void)row;
    if (!out || cap < 0)
        return -1;
    rt_res_entry tab[RT_RES_TABLE_MAX];
    int total = rt_res_table(gen, tab, RT_RES_TABLE_MAX);
    int n = 0;
    for (int i = 0; i < total; i++) {
        if (tab[i].port != port || tab[i].is_master != is_master)
            continue;
        if (n >= cap)
            return n;
        out[n++] = tab[i];
    }
    return n;
}

uint32_t __Runtime_res_reserved_pktid_mask(rt_res_gen gen) {
    rt_res_entry tab[RT_RES_TABLE_MAX];
    int total = rt_res_table(gen, tab, RT_RES_TABLE_MAX);
    uint32_t m = 0;
    for (int i = 0; i < total; i++) {
        if (tab[i].is_master)
            continue; /* masters carry no pkt-id */
        if (tab[i].pkt_id < 32)
            m |= (1u << tab[i].pkt_id);
    }
    return m;
}

uint32_t __Runtime_res_reserved_arbiter_mask(rt_res_gen gen) {
    rt_res_entry tab[RT_RES_TABLE_MAX];
    int total = rt_res_table(gen, tab, RT_RES_TABLE_MAX);
    uint32_t m = 0;
    for (int i = 0; i < total; i++)
        if (tab[i].arbiter < 32)
            m |= (1u << tab[i].arbiter);
    return m;
}

int __Runtime_res_reserved_slot_mask(rt_res_gen gen, uint8_t port, uint8_t is_master) {
    if (is_master)
        return 0; /* masters have no slave slots */
    rt_res_entry tab[RT_RES_TABLE_MAX];
    int total = rt_res_table(gen, tab, RT_RES_TABLE_MAX);
    int m = 0;
    for (int i = 0; i < total; i++) {
        if (tab[i].is_master || tab[i].port != port)
            continue;
        if (tab[i].slot < 31)
            m |= (1 << tab[i].slot);
    }
    return m;
}

int __Runtime_res_is_reserved(rt_res_gen gen, uint8_t port, uint8_t is_master, uint8_t slot, uint8_t arbiter,
                              uint8_t msel, uint8_t pkt_id) {
    rt_res_entry tab[RT_RES_TABLE_MAX];
    int total = rt_res_table(gen, tab, RT_RES_TABLE_MAX);
    for (int i = 0; i < total; i++) {
        const rt_res_entry *e = &tab[i];
        if (e->port != port || e->is_master != is_master || e->arbiter != arbiter)
            continue;
        if (is_master) {
            /* @msel carries the emitted mselen bitmap. A master is reserved when
             * every mselen bit it pulls is within the reserved superset for that
             * port (heads legitimately pull subsets: 0x1/0x3/0x7 for return). */
            if ((msel & (uint8_t)~e->msel) == 0)
                return 1;
            continue;
        }
        if (e->slot != slot || e->msel != msel)
            continue;
        /* Slave slot: pkt_id must match the reserved value (accept-any return
         * slots carry pkt_id 0, matching the planner's pkt=0). */
        if (e->pkt_id == pkt_id)
            return 1;
    }
    return 0;
}

rt_res_gen __Runtime_res_gen_from_name(const char *name) {
    if (!name)
        return RT_RES_GEN2;
    /* Accept "Gen1"/"Gen2"/"Gen5" and bare "1"/"2"/"5" (last char is the digit). */
    char last = 0;
    for (const char *p = name; *p; p++)
        last = *p;
    switch (last) {
    case '1':
        return RT_RES_GEN1;
    case '5':
        return RT_RES_GEN5;
    case '2':
    default:
        return RT_RES_GEN2;
    }
}
