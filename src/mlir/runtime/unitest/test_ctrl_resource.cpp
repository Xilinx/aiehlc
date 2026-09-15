// Host unit tests for the control-plane reservation table (aie_runtime_resource.c).
#include <cassert>
#include <cstdio>
extern "C" {
#include "aie_runtime_resource.h"
}

// Count reserved slot entries on a slave port for the given fabric direction.
static int slot_count(rt_res_gen gen, uint8_t port, uint8_t is_ret) {
    rt_res_entry e[32];
    int n = __Runtime_res_port_reserved(gen, 0, 0, port, /*is_master=*/0, e, 32);
    int c = 0;
    for (int i = 0; i < n; i++)
        if (e[i].is_ret == is_ret)
            c++;
    return c;
}

// True if a master enable is reserved on this port for the given fabric.
static bool has_master(rt_res_gen gen, uint8_t port, uint8_t is_ret) {
    rt_res_entry e[32];
    int n = __Runtime_res_port_reserved(gen, 0, 0, port, /*is_master=*/1, e, 32);
    for (int i = 0; i < n; i++)
        if (e[i].is_ret == is_ret)
            return true;
    return false;
}

int main() {
    const rt_res_gen gen = RT_RES_GEN5;

    // ---- #1 list API: reserved slots per port ----
    // Forward ingress-slave slots (WEST/SOUTH): 4 CONSUME (rowidx 0..3) + BCAST +
    // TRANSIT_N = 6 forward slots each.
    assert(slot_count(gen, RT_RES_PORT_WEST, /*ret=*/0) == 6);
    assert(slot_count(gen, RT_RES_PORT_SOUTH, /*ret=*/0) == 6);
    // Return slave slots: one accept-any slot per return slave port.
    assert(slot_count(gen, RT_RES_PORT_CTRL, /*ret=*/1) == 1);  // RET_LOCAL
    assert(slot_count(gen, RT_RES_PORT_EAST, /*ret=*/1) == 1);  // RET_TRANSIT
    assert(slot_count(gen, RT_RES_PORT_NORTH, /*ret=*/1) == 1); // RET_NORTH

    // Forward masters: CTRL/EAST/NORTH. Return masters: WEST/SOUTH.
    assert(has_master(gen, RT_RES_PORT_CTRL, /*ret=*/0));
    assert(has_master(gen, RT_RES_PORT_EAST, /*ret=*/0));
    assert(has_master(gen, RT_RES_PORT_NORTH, /*ret=*/0));
    assert(has_master(gen, RT_RES_PORT_WEST, /*ret=*/1));
    assert(has_master(gen, RT_RES_PORT_SOUTH, /*ret=*/1));

    // ---- pkt-id mask == {0,1,2,3,16} ----
    uint32_t pmask = __Runtime_res_reserved_pktid_mask(gen);
    uint32_t expect_p = (1u << 0) | (1u << 1) | (1u << 2) | (1u << 3) | (1u << 16);
    assert(pmask == expect_p);

    // ---- arbiter mask == {0,1} ----
    uint32_t amask = __Runtime_res_reserved_arbiter_mask(gen);
    assert(amask == ((1u << 0) | (1u << 1)));

    // ---- per-port reserved slot mask ----
    // Forward ingress: slots {0,1,2} = 0x7.
    assert(__Runtime_res_reserved_slot_mask(gen, RT_RES_PORT_WEST, 0) == 0x7);
    // Return CTRL slave: slot 0 only.
    assert(__Runtime_res_reserved_slot_mask(gen, RT_RES_PORT_CTRL, 0) == 0x1);
    // A master has no slave slots.
    assert(__Runtime_res_reserved_slot_mask(gen, RT_RES_PORT_CTRL, 1) == 0);

    // ---- is_reserved: accept every op the planner emits ----
    // Forward CONSUME slot on WEST for each rowidx.
    for (uint8_t r = 0; r <= RT_RES_MAX_ROW_IDX; r++)
        assert(__Runtime_res_is_reserved(gen, RT_RES_PORT_WEST, 0, RT_RES_SLOT_CONSUME, RT_RES_ARB_CTRL,
                                         RT_RES_MSEL_CONSUME, r));
    // Forward BCAST + TRANSIT_N.
    assert(__Runtime_res_is_reserved(gen, RT_RES_PORT_SOUTH, 0, RT_RES_SLOT_BCAST, RT_RES_ARB_CTRL, RT_RES_MSEL_BCAST,
                                     RT_RES_ID_BCAST));
    assert(__Runtime_res_is_reserved(gen, RT_RES_PORT_WEST, 0, RT_RES_SLOT_TRANSIT_N, RT_RES_ARB_CTRL,
                                     RT_RES_MSEL_TRANSIT_N, RT_RES_ID_TRANSIT));
    // Forward masters (exact mselen the planner emits: CTRL 0x3, EAST 0xB, NORTH 0x6).
    assert(__Runtime_res_is_reserved(gen, RT_RES_PORT_CTRL, 1, RT_RES_NA, RT_RES_ARB_CTRL, 0x3, RT_RES_NA));
    assert(__Runtime_res_is_reserved(gen, RT_RES_PORT_EAST, 1, RT_RES_NA, RT_RES_ARB_CTRL, 0xB, RT_RES_NA));
    assert(__Runtime_res_is_reserved(gen, RT_RES_PORT_NORTH, 1, RT_RES_NA, RT_RES_ARB_CTRL, 0x6, RT_RES_NA));
    // Return slots (accept-any pkt=0).
    assert(__Runtime_res_is_reserved(gen, RT_RES_PORT_CTRL, 0, RT_RES_SLOT_RET_LOCAL, RT_RES_ARB_RET,
                                     RT_RES_MSEL_RET_LOCAL, 0));
    assert(__Runtime_res_is_reserved(gen, RT_RES_PORT_EAST, 0, RT_RES_SLOT_RET_TRANSIT, RT_RES_ARB_RET,
                                     RT_RES_MSEL_RET_TRANSIT, 0));
    // Return masters: heads pull subsets (0x1/0x3/0x7 on WEST/SOUTH).
    assert(__Runtime_res_is_reserved(gen, RT_RES_PORT_WEST, 1, RT_RES_NA, RT_RES_ARB_RET, 0x1, RT_RES_NA));
    assert(__Runtime_res_is_reserved(gen, RT_RES_PORT_WEST, 1, RT_RES_NA, RT_RES_ARB_RET, 0x3, RT_RES_NA));
    assert(__Runtime_res_is_reserved(gen, RT_RES_PORT_SOUTH, 1, RT_RES_NA, RT_RES_ARB_RET, 0x7, RT_RES_NA));

    // ---- is_reserved: reject a bogus tuple (slot 3, arbiter 2) ----
    assert(!__Runtime_res_is_reserved(gen, RT_RES_PORT_WEST, 0, /*slot=*/3, /*arb=*/2, /*msel=*/3, /*pkt=*/5));
    // A forward slot with a wrong pkt-id (e.g. 7) is not reserved.
    assert(!__Runtime_res_is_reserved(gen, RT_RES_PORT_WEST, 0, RT_RES_SLOT_CONSUME, RT_RES_ARB_CTRL,
                                      RT_RES_MSEL_CONSUME, 7));
    // A master mselen with a bit outside the reserved superset is rejected.
    assert(!__Runtime_res_is_reserved(gen, RT_RES_PORT_CTRL, 1, RT_RES_NA, RT_RES_ARB_CTRL, 0x1F, RT_RES_NA));

    // ---- gen-from-name helper ----
    assert(__Runtime_res_gen_from_name("Gen5") == RT_RES_GEN5);
    assert(__Runtime_res_gen_from_name("Gen1") == RT_RES_GEN1);
    assert(__Runtime_res_gen_from_name("Gen2") == RT_RES_GEN2);
    assert(__Runtime_res_gen_from_name("5") == RT_RES_GEN5);
    assert(__Runtime_res_gen_from_name(nullptr) == RT_RES_GEN2);

    printf("PASS\n");
    return 0;
}
