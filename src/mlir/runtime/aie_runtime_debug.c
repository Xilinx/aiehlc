/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

#include "aie_runtime_debug.h"
#include "aie_device_map.h"
#include <stdio.h>
#include <string.h>

/* --------------------------------------------------------------------------
 * AIE generation — global, default AIE2PS
 * -------------------------------------------------------------------------- */

AieRt_AieGen g_aiert_gen = AIERT_GEN_AIE2PS;

void AieRt_SetAieGen(AieRt_AieGen gen) { g_aiert_gen = gen; }

/* --------------------------------------------------------------------------
 * Event name lookup tables
 *
 * Sourced from AIE2PS architecture specification:
 *   doc/aie2pscoreevent.md  — Table 4-8 (Core module, IDs 0-127)
 *   doc/aie2psmemevent.md   — Mem module (IDs 0-127)
 *   doc/aie2psshimevent.md  — Table 7-12 (Shim/PL module, IDs 0-181)
 *
 * Tables are sparse arrays indexed by local hardware event ID.
 * NULL entries indicate gaps (reserved / not defined for this generation).
 * -------------------------------------------------------------------------- */

/* --- CORE module (AIE2PS, Table 4-8, max ID=127) --- */
static const char *s_core_evt_names[128] = {
    "NONE",                            /* 0 */
    "TRUE",                            /* 1 */
    "GROUP_0",                         /* 2 */
    "TIMER_SYNC",                      /* 3 */
    "TIMER_VALUE_REACHED",             /* 4 */
    "PERF_CNT_0",                      /* 5 */
    "PERF_CNT_1",                      /* 6 */
    "PERF_CNT_2",                      /* 7 */
    "PERF_CNT_3",                      /* 8 */
    "COMBO_EVENT_0",                   /* 9 */
    "COMBO_EVENT_1",                   /* 10 */
    "COMBO_EVENT_2",                   /* 11 */
    "COMBO_EVENT_3",                   /* 12 */
    "EDGE_DETECTION_EVENT_0",          /* 13 */
    "EDGE_DETECTION_EVENT_1",          /* 14 */
    "GROUP_PC_EVENT",                  /* 15 */
    "PC_0",                            /* 16 */
    "PC_1",                            /* 17 */
    "PC_2",                            /* 18 */
    "PC_3",                            /* 19 */
    "PC_RANGE_0_1",                    /* 20 */
    "PC_RANGE_2_3",                    /* 21 */
    "GROUP_CORE_STALL",                /* 22 */
    "MEMORY_STALL",                    /* 23 */
    "STREAM_STALL",                    /* 24 */
    "CASCADE_STALL",                   /* 25 */
    "LOCK_STALL",                      /* 26 */
    "DEBUG_HALTED",                    /* 27 */
    "ACTIVE",                          /* 28 */
    "DISABLED",                        /* 29 */
    "ECC_ERROR_STALL",                 /* 30 */
    "ECC_SCRUBBING_STALL",             /* 31 */
    "GROUP_CORE_PROGRAM_FLOW",         /* 32 */
    "INSTR_EVENT_0",                   /* 33 */
    "INSTR_EVENT_1",                   /* 34 */
    "INSTR_CALL",                      /* 35 */
    "INSTR_RETURN",                    /* 36 */
    "INSTR_VECTOR",                    /* 37 */
    "INSTR_LOAD",                      /* 38 */
    "INSTR_STORE",                     /* 39 */
    "INSTR_STREAM_GET",                /* 40 */
    "INSTR_STREAM_PUT",                /* 41 */
    "INSTR_CASCADE_GET",               /* 42 */
    "INSTR_CASCADE_PUT",               /* 43 */
    "INSTR_LOCK_ACQUIRE_REQ",          /* 44 */
    "INSTR_LOCK_RELEASE_REQ",          /* 45 */
    "GROUP_ERRORS_0",                  /* 46 */
    "GROUP_ERRORS_1",                  /* 47 */
    "SRS_OVERFLOW",                    /* 48 */
    "UPS_OVERFLOW",                    /* 49 */
    "FP_HUGE",                         /* 50 */
    "INT_FP_ZERO",                     /* 51 */
    "FP_INVALID",                      /* 52 */
    "FP_INF",                          /* 53 */
    NULL,                              /* 54 — reserved */
    "PM_REG_ACCESS_FAILURE",           /* 55 */
    "STREAM_PKT_PARITY_ERROR",         /* 56 */
    "CONTROL_PKT_ERROR",               /* 57 */
    "AXI_MM_SLAVE_ERROR",              /* 58 */
    "INSTR_DECOMPRESSION_ERROR",       /* 59 — deprecated, never fires */
    "DM_ADDRESS_OUT_OF_RANGE",         /* 60 */
    "PM_ECC_ERROR_SCRUB_CORRECTED",    /* 61 */
    "PM_ECC_ERROR_SCRUB_2BIT",         /* 62 */
    "PM_ECC_ERROR_1BIT",               /* 63 */
    "PM_ECC_ERROR_2BIT",               /* 64 */
    "PM_ADDRESS_OUT_OF_RANGE",         /* 65 */
    "DM_ACCESS_TO_UNAVAILABLE",        /* 66 */
    "LOCK_ACCESS_TO_UNAVAILABLE",      /* 67 */
    "INSTR_WARNING",                   /* 68 */
    "INSTR_ERROR",                     /* 69 */
    "SPARSITY_OVERFLOW",               /* 70 */
    "STREAM_SWITCH_PORT_PARITY_ERROR", /* 71 */
    "PROCESSOR_BUS_ERROR",             /* 72 */
    "GROUP_STREAM_SWITCH",             /* 73 */
    "PORT_IDLE_0",                     /* 74 */
    "PORT_RUNNING_0",                  /* 75 */
    "PORT_STALLED_0",                  /* 76 */
    "PORT_TLAST_0",                    /* 77 */
    "PORT_IDLE_1",                     /* 78 */
    "PORT_RUNNING_1",                  /* 79 */
    "PORT_STALLED_1",                  /* 80 */
    "PORT_TLAST_1",                    /* 81 */
    "PORT_IDLE_2",                     /* 82 */
    "PORT_RUNNING_2",                  /* 83 */
    "PORT_STALLED_2",                  /* 84 */
    "PORT_TLAST_2",                    /* 85 */
    "PORT_IDLE_3",                     /* 86 */
    "PORT_RUNNING_3",                  /* 87 */
    "PORT_STALLED_3",                  /* 88 */
    "PORT_TLAST_3",                    /* 89 */
    "PORT_IDLE_4",                     /* 90 */
    "PORT_RUNNING_4",                  /* 91 */
    "PORT_STALLED_4",                  /* 92 */
    "PORT_TLAST_4",                    /* 93 */
    "PORT_IDLE_5",                     /* 94 */
    "PORT_RUNNING_5",                  /* 95 */
    "PORT_STALLED_5",                  /* 96 */
    "PORT_TLAST_5",                    /* 97 */
    "PORT_IDLE_6",                     /* 98 */
    "PORT_RUNNING_6",                  /* 99 */
    "PORT_STALLED_6",                  /* 100 */
    "PORT_TLAST_6",                    /* 101 */
    "PORT_IDLE_7",                     /* 102 */
    "PORT_RUNNING_7",                  /* 103 */
    "PORT_STALLED_7",                  /* 104 */
    "PORT_TLAST_7",                    /* 105 */
    "GROUP_BROADCAST",                 /* 106 */
    "BROADCAST_0",                     /* 107 */
    "BROADCAST_1",                     /* 108 */
    "BROADCAST_2",                     /* 109 */
    "BROADCAST_3",                     /* 110 */
    "BROADCAST_4",                     /* 111 */
    "BROADCAST_5",                     /* 112 */
    "BROADCAST_6",                     /* 113 */
    "BROADCAST_7",                     /* 114 */
    "BROADCAST_8",                     /* 115 */
    "BROADCAST_9",                     /* 116 */
    "BROADCAST_10",                    /* 117 */
    "BROADCAST_11",                    /* 118 */
    "BROADCAST_12",                    /* 119 */
    "BROADCAST_13",                    /* 120 */
    "BROADCAST_14",                    /* 121 */
    "BROADCAST_15",                    /* 122 */
    "GROUP_USER_EVENT",                /* 123 */
    "USER_EVENT_0",                    /* 124 */
    "USER_EVENT_1",                    /* 125 */
    "USER_EVENT_2",                    /* 126 */
    "USER_EVENT_3",                    /* 127 */
};
#define AIERT_CORE_EVT_MAX 127u

/* --- MEM module (AIE2PS, aie2psmemevent.md, max ID=127) --- */
static const char *s_mem_evt_names[128] = {
    "NONE",                           /* 0 */
    "TRUE",                           /* 1 */
    "GROUP_0",                        /* 2 */
    "TIMER_SYNC",                     /* 3 */
    "TIMER_VALUE_REACHED",            /* 4 */
    "PERF_CNT0_EVENT",                /* 5 */
    "PERF_CNT1_EVENT",                /* 6 */
    "COMBO_EVENT_0",                  /* 7 */
    "COMBO_EVENT_1",                  /* 8 */
    "COMBO_EVENT_2",                  /* 9 */
    "COMBO_EVENT_3",                  /* 10 */
    "EDGE_DETECTION_EVENT_0",         /* 11 */
    "EDGE_DETECTION_EVENT_1",         /* 12 */
    NULL,                             /* 13 — reserved */
    NULL,                             /* 14 — reserved */
    "GROUP_WATCHPOINT",               /* 15 */
    "WATCHPOINT_0",                   /* 16 */
    "WATCHPOINT_1",                   /* 17 */
    "GROUP_DMA_ACTIVITY",             /* 18 */
    "DMA_S2MM_0_START_TASK",          /* 19 */
    "DMA_S2MM_1_START_TASK",          /* 20 */
    "DMA_MM2S_0_START_TASK",          /* 21 */
    "DMA_MM2S_1_START_TASK",          /* 22 */
    "DMA_S2MM_0_FINISHED_BD",         /* 23 */
    "DMA_S2MM_1_FINISHED_BD",         /* 24 */
    "DMA_MM2S_0_FINISHED_BD",         /* 25 */
    "DMA_MM2S_1_FINISHED_BD",         /* 26 */
    "DMA_S2MM_0_FINISHED_TASK",       /* 27 */
    "DMA_S2MM_1_FINISHED_TASK",       /* 28 */
    "DMA_MM2S_0_FINISHED_TASK",       /* 29 */
    "DMA_MM2S_1_FINISHED_TASK",       /* 30 */
    "DMA_S2MM_0_STALLED_LOCK",        /* 31 */
    "DMA_S2MM_1_STALLED_LOCK",        /* 32 */
    "DMA_MM2S_0_STALLED_LOCK",        /* 33 */
    "DMA_MM2S_1_STALLED_LOCK",        /* 34 */
    "DMA_S2MM_0_STREAM_STARVATION",   /* 35 */
    "DMA_S2MM_1_STREAM_STARVATION",   /* 36 */
    "DMA_MM2S_0_STREAM_BACKPRESSURE", /* 37 */
    "DMA_MM2S_1_STREAM_BACKPRESSURE", /* 38 */
    "DMA_S2MM_0_MEMORY_BACKPRESSURE", /* 39 */
    "DMA_S2MM_1_MEMORY_BACKPRESSURE", /* 40 */
    "DMA_MM2S_0_MEMORY_STARVATION",   /* 41 */
    "DMA_MM2S_1_MEMORY_STARVATION",   /* 42 */
    "GROUP_LOCK",                     /* 43 */
    "LOCK_SEL0_ACQ_EQ",               /* 44 */
    "LOCK_SEL0_ACQ_GE",               /* 45 */
    "LOCK_SEL0_REL",                  /* 46 */
    "LOCK_SEL0_EQUAL_TO_VALUE",       /* 47 */
    "LOCK_SEL1_ACQ_EQ",               /* 48 */
    "LOCK_SEL1_ACQ_GE",               /* 49 */
    "LOCK_SEL1_REL",                  /* 50 */
    "LOCK_SEL1_EQUAL_TO_VALUE",       /* 51 */
    "LOCK_SEL2_ACQ_EQ",               /* 52 */
    "LOCK_SEL2_ACQ_GE",               /* 53 */
    "LOCK_SEL2_REL",                  /* 54 */
    "LOCK_SEL2_EQUAL_TO_VALUE",       /* 55 */
    "LOCK_SEL3_ACQ_EQ",               /* 56 */
    "LOCK_SEL3_ACQ_GE",               /* 57 */
    "LOCK_SEL3_REL",                  /* 58 */
    "LOCK_SEL3_EQUAL_TO_VALUE",       /* 59 */
    "LOCK_SEL4_ACQ_EQ",               /* 60 */
    "LOCK_SEL4_ACQ_GE",               /* 61 */
    "LOCK_SEL4_REL",                  /* 62 */
    "LOCK_SEL4_EQUAL_TO_VALUE",       /* 63 */
    "LOCK_SEL5_ACQ_EQ",               /* 64 */
    "LOCK_SEL5_ACQ_GE",               /* 65 */
    "LOCK_SEL5_REL",                  /* 66 */
    "LOCK_SEL5_EQUAL_TO_VALUE",       /* 67 */
    "LOCK_SEL6_ACQ_EQ",               /* 68 */
    "LOCK_SEL6_ACQ_GE",               /* 69 */
    "LOCK_SEL6_REL",                  /* 70 */
    "LOCK_SEL6_EQUAL_TO_VALUE",       /* 71 */
    "LOCK_SEL7_ACQ_EQ",               /* 72 */
    "LOCK_SEL7_ACQ_GE",               /* 73 */
    "LOCK_SEL7_REL",                  /* 74 */
    "LOCK_SEL7_EQUAL_TO_VALUE",       /* 75 */
    "GROUP_MEMORY_CONFLICT",          /* 76 */
    "CONFLICT_DM_BANK_0",             /* 77 */
    "CONFLICT_DM_BANK_1",             /* 78 */
    "CONFLICT_DM_BANK_2",             /* 79 */
    "CONFLICT_DM_BANK_3",             /* 80 */
    "CONFLICT_DM_BANK_4",             /* 81 */
    "CONFLICT_DM_BANK_5",             /* 82 */
    "CONFLICT_DM_BANK_6",             /* 83 */
    "CONFLICT_DM_BANK_7",             /* 84 */
    NULL,                             /* 85 — reserved */
    "GROUP_ERRORS",                   /* 86 */
    "DM_ECC_ERROR_SCRUB_CORRECTED",   /* 87 */
    "DM_ECC_ERROR_SCRUB_2BIT",        /* 88 */
    "DM_ECC_ERROR_1BIT",              /* 89 */
    "DM_ECC_ERROR_2BIT",              /* 90 */
    "DM_PARITY_ERROR_BANK_2",         /* 91 */
    "DM_PARITY_ERROR_BANK_3",         /* 92 */
    "DM_PARITY_ERROR_BANK_4",         /* 93 */
    "DM_PARITY_ERROR_BANK_5",         /* 94 */
    "DM_PARITY_ERROR_BANK_6",         /* 95 */
    "DM_PARITY_ERROR_BANK_7",         /* 96 */
    "DMA_S2MM_0_ERROR",               /* 97 */
    "DMA_S2MM_1_ERROR",               /* 98 */
    "DMA_MM2S_0_ERROR",               /* 99 */
    "DMA_MM2S_1_ERROR",               /* 100 */
    "LOCK_ERROR",                     /* 101 */
    "DMA_TASK_TOKEN_STALL",           /* 102 */
    NULL,                             /* 103 — reserved */
    NULL,                             /* 104 — reserved */
    NULL,                             /* 105 — reserved */
    "GROUP_BROADCAST",                /* 106 */
    "BROADCAST_0",                    /* 107 */
    "BROADCAST_1",                    /* 108 */
    "BROADCAST_2",                    /* 109 */
    "BROADCAST_3",                    /* 110 */
    "BROADCAST_4",                    /* 111 */
    "BROADCAST_5",                    /* 112 */
    "BROADCAST_6",                    /* 113 */
    "BROADCAST_7",                    /* 114 */
    "BROADCAST_8",                    /* 115 */
    "BROADCAST_9",                    /* 116 */
    "BROADCAST_10",                   /* 117 */
    "BROADCAST_11",                   /* 118 */
    "BROADCAST_12",                   /* 119 */
    "BROADCAST_13",                   /* 120 */
    "BROADCAST_14",                   /* 121 */
    "BROADCAST_15",                   /* 122 */
    "GROUP_USER_EVENT",               /* 123 */
    "USER_EVENT_0",                   /* 124 */
    "USER_EVENT_1",                   /* 125 */
    "USER_EVENT_2",                   /* 126 */
    "USER_EVENT_3",                   /* 127 */
};
#define AIERT_MEM_EVT_MAX 127u

/* --- PL module (Shim tile) event status register bit layout.
 *
 * Source: aie2psshimevent.md Table 7-12 "Shim Event List" (AIE2PS)
 * This is the HARDWARE event status register bit layout, NOT the XAie
 * driver enum values (xaie_events_aieml.h / xaie_events_aie2ps.h).
 * The XAie enums are software IDs used to program event counters and
 * do NOT match the hardware register bit positions.
 *
 * Max hardware event ID = 181 (USER_EVENT_1). Events 182-234 are uC
 * module events (Table 8-7) and not included here.
 */
static const char *s_pl_evt_names[182] = {
    "NONE",                                  /* 0 */
    "TRUE",                                  /* 1 */
    "GROUP_0",                               /* 2 */
    "TIMER_SYNC",                            /* 3 */
    "TIMER_VALUE_REACHED",                   /* 4 */
    "PERF_CNT0_EVENT",                       /* 5 */
    "PERF_CNT1_EVENT",                       /* 6 */
    "COMBO_EVENT_0",                         /* 7 */
    "COMBO_EVENT_1",                         /* 8 */
    "COMBO_EVENT_2",                         /* 9 */
    "COMBO_EVENT_3",                         /* 10 */
    "EDGE_DETECTION_EVENT_0",                /* 11 */
    "EDGE_DETECTION_EVENT_1",                /* 12 */
    "GROUP_NOC_0_DMA_ACTIVITY",              /* 13 */
    "NOC_0_DMA_S2MM_0_START_TASK",           /* 14 */
    "NOC_0_DMA_S2MM_1_START_TASK",           /* 15 */
    "NOC_0_DMA_MM2S_0_START_TASK",           /* 16 */
    "NOC_0_DMA_MM2S_1_START_TASK",           /* 17 */
    "NOC_0_DMA_S2MM_0_FINISHED_BD",          /* 18 */
    "NOC_0_DMA_S2MM_1_FINISHED_BD",          /* 19 */
    "NOC_0_DMA_MM2S_0_FINISHED_BD",          /* 20 */
    "NOC_0_DMA_MM2S_1_FINISHED_BD",          /* 21 */
    "NOC_0_DMA_S2MM_0_FINISHED_TASK",        /* 22 */
    "NOC_0_DMA_S2MM_1_FINISHED_TASK",        /* 23 */
    "NOC_0_DMA_MM2S_0_FINISHED_TASK",        /* 24 */
    "NOC_0_DMA_MM2S_1_FINISHED_TASK",        /* 25 */
    "NOC_0_DMA_S2MM_0_STALLED_LOCK",         /* 26 */
    "NOC_0_DMA_S2MM_1_STALLED_LOCK",         /* 27 */
    "NOC_0_DMA_MM2S_0_STALLED_LOCK",         /* 28 */
    "NOC_0_DMA_MM2S_1_STALLED_LOCK",         /* 29 */
    "NOC_0_DMA_S2MM_0_STREAM_STARVATION",    /* 30 */
    "NOC_0_DMA_S2MM_1_STREAM_STARVATION",    /* 31 */
    "NOC_0_DMA_MM2S_0_STREAM_BACKPRESSURE",  /* 32 */
    "NOC_0_DMA_MM2S_1_STREAM_BACKPRESSURE",  /* 33 */
    "NOC_0_DMA_S2MM_0_MEMORY_BACKPRESSURE",  /* 34 */
    "NOC_0_DMA_S2MM_1_MEMORY_BACKPRESSURE",  /* 35 */
    "NOC_0_DMA_MM2S_0_MEMORY_STARVATION",    /* 36 */
    "NOC_0_DMA_MM2S_1_MEMORY_STARVATION",    /* 37 */
    "GROUP_NOC_1_DMA_ACTIVITY",              /* 38 */
    "NOC_1_DMA_S2MM_0_START_TASK",           /* 39 */
    "NOC_1_DMA_S2MM_1_START_TASK",           /* 40 */
    "NOC_1_DMA_MM2S_0_START_TASK",           /* 41 */
    "NOC_1_DMA_MM2S_1_START_TASK",           /* 42 */
    "NOC_1_DMA_S2MM_0_FINISHED_BD",          /* 43 */
    "NOC_1_DMA_S2MM_1_FINISHED_BD",          /* 44 */
    "NOC_1_DMA_MM2S_0_FINISHED_BD",          /* 45 */
    "NOC_1_DMA_MM2S_1_FINISHED_BD",          /* 46 */
    "NOC_1_DMA_S2MM_0_FINISHED_TASK",        /* 47 */
    "NOC_1_DMA_S2MM_1_FINISHED_TASK",        /* 48 */
    "NOC_1_DMA_MM2S_0_FINISHED_TASK",        /* 49 */
    "NOC_1_DMA_MM2S_1_FINISHED_TASK",        /* 50 */
    "NOC_1_DMA_S2MM_0_STALLED_LOCK",         /* 51 */
    "NOC_1_DMA_S2MM_1_STALLED_LOCK",         /* 52 */
    "NOC_1_DMA_MM2S_0_STALLED_LOCK",         /* 53 */
    "NOC_1_DMA_MM2S_1_STALLED_LOCK",         /* 54 */
    "NOC_1_DMA_S2MM_0_STREAM_STARVATION",    /* 55 */
    "NOC_1_DMA_S2MM_1_STREAM_STARVATION",    /* 56 */
    "NOC_1_DMA_MM2S_0_STREAM_BACKPRESSURE",  /* 57 */
    "NOC_1_DMA_MM2S_1_STREAM_BACKPRESSURE",  /* 58 */
    "NOC_1_DMA_S2MM_0_MEMORY_BACKPRESSURE",  /* 59 */
    "NOC_1_DMA_S2MM_1_MEMORY_BACKPRESSURE",  /* 60 */
    "NOC_1_DMA_MM2S_0_MEMORY_STARVATION",    /* 61 */
    "NOC_1_DMA_MM2S_1_MEMORY_STARVATION",    /* 62 */
    "GROUP_NOC_0_LOCK",                      /* 63 */
    "NOC_0_LOCK_0_ACQ_EQ",                   /* 64 */
    "NOC_0_LOCK_0_ACQ_GE",                   /* 65 */
    "NOC_0_LOCK_0_REL",                      /* 66 */
    "NOC_0_LOCK_0_EQUAL_TO_VALUE",           /* 67 */
    "NOC_0_LOCK_1_ACQ_EQ",                   /* 68 */
    "NOC_0_LOCK_1_ACQ_GE",                   /* 69 */
    "NOC_0_LOCK_1_REL",                      /* 70 */
    "NOC_0_LOCK_1_EQUAL_TO_VALUE",           /* 71 */
    "NOC_0_LOCK_2_ACQ_EQ",                   /* 72 */
    "NOC_0_LOCK_2_ACQ_GE",                   /* 73 */
    "NOC_0_LOCK_2_REL",                      /* 74 */
    "NOC_0_LOCK_2_EQUAL_TO_VALUE",           /* 75 */
    "NOC_0_LOCK_3_ACQ_EQ",                   /* 76 */
    "NOC_0_LOCK_3_ACQ_GE",                   /* 77 */
    "NOC_0_LOCK_3_REL",                      /* 78 */
    "NOC_0_LOCK_3_EQUAL_TO_VALUE",           /* 79 */
    "NOC_0_LOCK_4_ACQ_EQ",                   /* 80 */
    "NOC_0_LOCK_4_ACQ_GE",                   /* 81 */
    "NOC_0_LOCK_4_REL",                      /* 82 */
    "NOC_0_LOCK_4_EQUAL_TO_VALUE",           /* 83 */
    "NOC_0_LOCK_5_ACQ_EQ",                   /* 84 */
    "NOC_0_LOCK_5_ACQ_GE",                   /* 85 */
    "NOC_0_LOCK_5_REL",                      /* 86 */
    "NOC_0_LOCK_5_EQUAL_TO_VALUE",           /* 87 */
    "GROUP_NOC_1_LOCK",                      /* 88 */
    "NOC_1_LOCK_0_ACQ_EQ",                   /* 89 */
    "NOC_1_LOCK_0_ACQ_GE",                   /* 90 */
    "NOC_1_LOCK_0_REL",                      /* 91 */
    "NOC_1_LOCK_0_EQUAL_TO_VALUE",           /* 92 */
    "NOC_1_LOCK_1_ACQ_EQ",                   /* 93 */
    "NOC_1_LOCK_1_ACQ_GE",                   /* 94 */
    "NOC_1_LOCK_1_REL",                      /* 95 */
    "NOC_1_LOCK_1_EQUAL_TO_VALUE",           /* 96 */
    "NOC_1_LOCK_2_ACQ_EQ",                   /* 97 */
    "NOC_1_LOCK_2_ACQ_GE",                   /* 98 */
    "NOC_1_LOCK_2_REL",                      /* 99 */
    "NOC_1_LOCK_2_EQUAL_TO_VALUE",           /* 100 */
    "NOC_1_LOCK_3_ACQ_EQ",                   /* 101 */
    "NOC_1_LOCK_3_ACQ_GE",                   /* 102 */
    "NOC_1_LOCK_3_REL",                      /* 103 */
    "NOC_1_LOCK_3_EQUAL_TO_VALUE",           /* 104 */
    "NOC_1_LOCK_4_ACQ_EQ",                   /* 105 */
    "NOC_1_LOCK_4_ACQ_GE",                   /* 106 */
    "NOC_1_LOCK_4_REL",                      /* 107 */
    "NOC_1_LOCK_4_EQUAL_TO_VALUE",           /* 108 */
    "NOC_1_LOCK_5_ACQ_EQ",                   /* 109 */
    "NOC_1_LOCK_5_ACQ_GE",                   /* 110 */
    "NOC_1_LOCK_5_REL",                      /* 111 */
    "NOC_1_LOCK_5_EQUAL_TO_VALUE",           /* 112 */
    "GROUP_ERRORS",                          /* 113 */
    "AXI_MM_SLAVE_TILE_ERROR",               /* 114 */
    "CONTROL_PKT_ERROR",                     /* 115 */
    "STREAM_SWITCH_PARITY_ERROR",            /* 116 */
    "AXI_MM_DECODE_NSU_ERROR",               /* 117 */
    "AXI_MM_SLAVE_NSU_ERROR",                /* 118 */
    "AXI_MM_UNSUPPORTED_TRAFFIC",            /* 119 */
    "AXI_MM_UNSECURE_ACCESS_IN_SECURE_MODE", /* 120 */
    "AXI_MM_BYTE_STROBE_ERROR",              /* 121 */
    "NOC_0_DMA_S2MM_ERROR",                  /* 122 */
    "NOC_1_DMA_S2MM_ERROR",                  /* 123 */
    "NOC_0_DMA_MM2S_ERROR",                  /* 124 */
    "NOC_1_DMA_MM2S_ERROR",                  /* 125 */
    "NOC_0_LOCK_ERROR",                      /* 126 */
    "NOC_1_LOCK_ERROR",                      /* 127 */
    "NOC_0_DMA_TASK_TOKEN_STALL",            /* 128 */
    "NOC_1_DMA_TASK_TOKEN_STALL",            /* 129 */
    "GROUP_STREAM_SWITCH",                   /* 130 */
    "PORT_IDLE_0",                           /* 131 */
    "PORT_RUNNING_0",                        /* 132 */
    "PORT_STALLED_0",                        /* 133 */
    "PORT_TLAST_0",                          /* 134 */
    "PORT_IDLE_1",                           /* 135 */
    "PORT_RUNNING_1",                        /* 136 */
    "PORT_STALLED_1",                        /* 137 */
    "PORT_TLAST_1",                          /* 138 */
    "PORT_IDLE_2",                           /* 139 */
    "PORT_RUNNING_2",                        /* 140 */
    "PORT_STALLED_2",                        /* 141 */
    "PORT_TLAST_2",                          /* 142 */
    "PORT_IDLE_3",                           /* 143 */
    "PORT_RUNNING_3",                        /* 144 */
    "PORT_STALLED_3",                        /* 145 */
    "PORT_TLAST_3",                          /* 146 */
    "PORT_IDLE_4",                           /* 147 */
    "PORT_RUNNING_4",                        /* 148 */
    "PORT_STALLED_4",                        /* 149 */
    "PORT_TLAST_4",                          /* 150 */
    "PORT_IDLE_5",                           /* 151 */
    "PORT_RUNNING_5",                        /* 152 */
    "PORT_STALLED_5",                        /* 153 */
    "PORT_TLAST_5",                          /* 154 */
    "PORT_IDLE_6",                           /* 155 */
    "PORT_RUNNING_6",                        /* 156 */
    "PORT_STALLED_6",                        /* 157 */
    "PORT_TLAST_6",                          /* 158 */
    "PORT_IDLE_7",                           /* 159 */
    "PORT_RUNNING_7",                        /* 160 */
    "PORT_STALLED_7",                        /* 161 */
    "PORT_TLAST_7",                          /* 162 */
    "GROUP_BROADCAST_A",                     /* 163 */
    "BROADCAST_A_0",                         /* 164 */
    "BROADCAST_A_1",                         /* 165 */
    "BROADCAST_A_2",                         /* 166 */
    "BROADCAST_A_3",                         /* 167 */
    "BROADCAST_A_4",                         /* 168 */
    "BROADCAST_A_5",                         /* 169 */
    "BROADCAST_A_6",                         /* 170 */
    "BROADCAST_A_7",                         /* 171 */
    "BROADCAST_A_8",                         /* 172 */
    "BROADCAST_A_9",                         /* 173 */
    "BROADCAST_A_10",                        /* 174 */
    "BROADCAST_A_11",                        /* 175 */
    "BROADCAST_A_12",                        /* 176 */
    "BROADCAST_A_13",                        /* 177 */
    "BROADCAST_A_14",                        /* 178 */
    "BROADCAST_A_15",                        /* 179 */
    "USER_EVENT_0",                          /* 180 */
    "USER_EVENT_1",                          /* 181 */
};
#define AIERT_PL_EVT_MAX 181u
#define AIERT_PL_EVT_AIE2PS_MAX AIERT_PL_EVT_MAX

/* --------------------------------------------------------------------------
 * Internal helpers
 * -------------------------------------------------------------------------- */

static int s_is_shim(XAie_LocType tile) { return tile.Row == XAIE_SHIM_ROW; }
static int s_is_memtile(XAie_LocType tile) {
    return tile.Row >= XAIE_RES_TILE_ROW_START && tile.Row < (XAIE_RES_TILE_ROW_START + XAIE_RES_TILE_NUM_ROWS);
}
static int s_is_core(XAie_LocType tile) { return tile.Row >= XAIE_AIE_TILE_ROW_START; }

static const char *s_dir_str(XAie_DmaDirection dir) { return (dir == DMA_S2MM) ? "S2MM" : "MM2S"; }

static const char *s_status_str(uint8_t status) {
    switch (status) {
    case AIERT_DMA_STATUS_IDLE:
        return "Idle";
    case AIERT_DMA_STATUS_RUNNING:
        return "Running";
    case AIERT_DMA_STATUS_PAUSED:
        return "Paused";
    default:
        return "Unknown";
    }
}
/* --------------------------------------------------------------------------
 * AieRt_EventName — module-local ID → string
 * -------------------------------------------------------------------------- */

const char *AieRt_EventName(XAie_ModuleType module, uint32_t id) {
    static char s_fallback[24]; /* non-re-entrant fallback for unknown IDs */

    if (module == XAIE_CORE_MOD) {
        if (id <= AIERT_CORE_EVT_MAX && s_core_evt_names[id])
            return s_core_evt_names[id];
    } else if (module == XAIE_MEM_MOD) {
        if (id <= AIERT_MEM_EVT_MAX && s_mem_evt_names[id])
            return s_mem_evt_names[id];
    } else if (module == XAIE_PL_MOD) {
        if (id <= AIERT_PL_EVT_MAX && s_pl_evt_names[id])
            return s_pl_evt_names[id];
    }
    /* Unknown: format into static buffer */
    snprintf(s_fallback, sizeof(s_fallback), "EVT_%u", (unsigned)id);
    return s_fallback;
}

/* --------------------------------------------------------------------------
 * Memory Module event status print — all set bits with names
 *
 * Reads the core tile Memory Module event status registers:
 *   0x00014200 : MEM module event IDs   0-31
 *   0x00014204 : MEM module event IDs  32-63
 *   0x00014208 : MEM module event IDs  64-95
 *   0x0001420C : MEM module event IDs  96-127
 *
 * For each set bit, prints: id + name from AieRt_EventName(XAIE_MEM_MOD, id)
 * -------------------------------------------------------------------------- */

void AieRt_PrintMemModuleEvents(XAie_DevInst *dev, XAie_LocType tile) {
    if (!s_is_core(tile)) {
        printf("[AieRt_Debug] tile(%u,%u): not a core tile, skip mem module events\n", (unsigned)tile.Col,
               (unsigned)tile.Row);
        return;
    }

    u64 base = ((u64)tile.Col << 25) | ((u64)tile.Row << 20);
    u32 reg[4] = {0, 0, 0, 0};
    XAie_Read32(dev, base | AIERT_CORE_MEM_EVT_STATUS_REG, &reg[0]);
    XAie_Read32(dev, base | AIERT_CORE_MEM_EVT_STATUS_REG1, &reg[1]);
    XAie_Read32(dev, base | AIERT_CORE_MEM_EVT_STATUS_REG2, &reg[2]);
    XAie_Read32(dev, base | AIERT_CORE_MEM_EVT_STATUS_REG3, &reg[3]);

    const char *gen_str = (g_aiert_gen == AIERT_GEN_AIE2PS) ? "AIE2PS" : "AIEML";
    printf("[AieRt_Debug] tile(%u,%u) MEM module events [gen=%s]"
           "  reg0=0x%08X  reg1=0x%08X  reg2=0x%08X  reg3=0x%08X\n",
           (unsigned)tile.Col, (unsigned)tile.Row, gen_str, (unsigned)reg[0], (unsigned)reg[1], (unsigned)reg[2],
           (unsigned)reg[3]);

    int any = 0;
    for (uint32_t r = 0; r < 4u; r++) {
        for (int bit = 0; bit < 32; bit++) {
            if (!((reg[r] >> bit) & 1u))
                continue;
            uint32_t id = r * 32u + (uint32_t)bit;
            if (id > AIERT_MEM_EVT_MAX)
                continue;
            printf("  MEM evt %3u  %s\n", (unsigned)id, AieRt_EventName(XAIE_MEM_MOD, id));
            any = 1;
        }
    }
    if (!any)
        printf("  (no MEM module events set)\n");
}

void AieRt_PrintMemModuleEventsAll(XAie_DevInst *dev, const XAie_LocType *tiles, uint32_t num_tiles) {
    const char *gen_str = (g_aiert_gen == AIERT_GEN_AIE2PS) ? "AIE2PS" : "AIEML";
    printf("[AieRt_Debug] ===== MEM Module Events [gen=%s] (%u tiles) =====\n", gen_str, (unsigned)num_tiles);
    for (uint32_t i = 0; i < num_tiles; i++) {
        if (s_is_core(tiles[i]))
            AieRt_PrintMemModuleEvents(dev, tiles[i]);
    }
}

/* --------------------------------------------------------------------------
 * Core Module event status print — all set bits with names
 *
 * Reads the core tile Core Module event status registers:
 *   0x00034200 : CORE module event IDs   0-31
 *   0x00034204 : CORE module event IDs  32-63
 *   0x00034208 : CORE module event IDs  64-95
 *   0x0003420C : CORE module event IDs  96-127
 *
 * For each set bit, prints: id + name from AieRt_EventName(XAIE_CORE_MOD, id)
 * -------------------------------------------------------------------------- */

void AieRt_PrintCoreModuleEvents(XAie_DevInst *dev, XAie_LocType tile) {
    if (!s_is_core(tile)) {
        printf("[AieRt_Debug] tile(%u,%u): not a core tile, skip core module events\n", (unsigned)tile.Col,
               (unsigned)tile.Row);
        return;
    }

    u64 base = ((u64)tile.Col << 25) | ((u64)tile.Row << 20);
    u32 reg[4] = {0, 0, 0, 0};
    XAie_Read32(dev, base | AIERT_CORE_CORE_EVT_STATUS_REG, &reg[0]);
    XAie_Read32(dev, base | AIERT_CORE_CORE_EVT_STATUS_REG1, &reg[1]);
    XAie_Read32(dev, base | AIERT_CORE_CORE_EVT_STATUS_REG2, &reg[2]);
    XAie_Read32(dev, base | AIERT_CORE_CORE_EVT_STATUS_REG3, &reg[3]);

    const char *gen_str = (g_aiert_gen == AIERT_GEN_AIE2PS) ? "AIE2PS" : "AIEML";
    printf("[AieRt_Debug] tile(%u,%u) CORE module events [gen=%s]"
           "  reg0=0x%08X  reg1=0x%08X  reg2=0x%08X  reg3=0x%08X\n",
           (unsigned)tile.Col, (unsigned)tile.Row, gen_str, (unsigned)reg[0], (unsigned)reg[1], (unsigned)reg[2],
           (unsigned)reg[3]);

    int any = 0;
    for (uint32_t r = 0; r < 4u; r++) {
        for (int bit = 0; bit < 32; bit++) {
            if (!((reg[r] >> bit) & 1u))
                continue;
            uint32_t id = r * 32u + (uint32_t)bit;
            if (id > AIERT_CORE_EVT_MAX)
                continue;
            printf("  CORE evt %3u  %s\n", (unsigned)id, AieRt_EventName(XAIE_CORE_MOD, id));
            any = 1;
        }
    }
    if (!any)
        printf("  (no CORE module events set)\n");
}

void AieRt_PrintCoreModuleEventsAll(XAie_DevInst *dev, const XAie_LocType *tiles, uint32_t num_tiles) {
    const char *gen_str = (g_aiert_gen == AIERT_GEN_AIE2PS) ? "AIE2PS" : "AIEML";
    printf("[AieRt_Debug] ===== CORE Module Events [gen=%s] (%u tiles) =====\n", gen_str, (unsigned)num_tiles);
    for (uint32_t i = 0; i < num_tiles; i++) {
        if (s_is_core(tiles[i]))
            AieRt_PrintCoreModuleEvents(dev, tiles[i]);
    }
}

/* --------------------------------------------------------------------------
 * Core status
 * -------------------------------------------------------------------------- */

void AieRt_PrintCoreStatus(XAie_DevInst *dev, XAie_LocType tile) {
    if (!s_is_core(tile)) {
        printf("[AieRt_Debug] tile(%u,%u) is not a core tile, skip core status\n", (unsigned)tile.Col,
               (unsigned)tile.Row);
        return;
    }

    u32 core_status = 0;
    AieRC rc = XAie_CoreGetStatus(dev, tile, &core_status);
    if (rc != XAIE_OK) {
        printf("[AieRt_Debug] CoreGetStatus tile(%u,%u) failed rc=%d\n", (unsigned)tile.Col, (unsigned)tile.Row,
               (int)rc);
        return;
    }

    u32 lr = 0;
    int in_reset = (core_status & XAIE_CORE_STATUS_RESET) != 0;
    if (!in_reset)
        XAie_CoreGetLRValue(dev, tile, &lr);

    /* Read status, PC, SP directly from hardware registers */
    u64 status_reg_off = ((u64)tile.Col << 25) | ((u64)tile.Row << 20) | AIERT_CORE_STATUS_REG;
    u64 pc_reg_off = ((u64)tile.Col << 25) | ((u64)tile.Row << 20) | AIERT_CORE_PC_REG;
    u64 sp_reg_off = ((u64)tile.Col << 25) | ((u64)tile.Row << 20) | AIERT_CORE_SP_REG;
    u32 raw_status = 0, raw_pc = 0, raw_sp = 0;
    XAie_Read32(dev, status_reg_off, &raw_status);
    XAie_Read32(dev, pc_reg_off, &raw_pc);
    XAie_Read32(dev, sp_reg_off, &raw_sp);

    /* Decode raw_status per xaiemlgbl_params.h / Core_Status register layout:
     *   [21] CORE_PROCESSOR_BUS_STALL   [20] CORE_DONE
     *   [19] ERROR_HALT                 [18] ECC_SCRUBBING_STALL
     *   [17] ECC_ERROR_STALL            [16] DEBUG_HALT
     *   [15] CASCADE_STALL_MCD         [14] CASCADE_STALL_SCD
     *   [12] STREAM_STALL_MS0          [10] STREAM_STALL_SS0
     *   [ 9] LOCK_STALL_E  [ 8] LOCK_STALL_N  [ 7] LOCK_STALL_W  [ 6] LOCK_STALL_S
     *   [ 5] MEMORY_STALL_E [ 4] MEMORY_STALL_N [ 3] MEMORY_STALL_W [ 2] MEMORY_STALL_S
     *   [ 1] RESET          [ 0] ENABLE
     * mask 0x003FFFFF  (bits 22-31 reserved)
     */
    printf("[AieRt_Debug] Core tile(%u,%u) status=0x%08X\n", (unsigned)tile.Col, (unsigned)tile.Row,
           (unsigned)core_status);
    printf("  Core_Status  : 0x%08X  (reg 0x%05X, mask 0x003FFFFF)\n", (unsigned)raw_status,
           (unsigned)AIERT_CORE_STATUS_REG);
    printf("  Core_PC      : 0x%05X  (reg 0x%05X, mask 0x000FFFFF)\n", (unsigned)(raw_pc & AIERT_CORE_PC_MASK),
           (unsigned)AIERT_CORE_PC_REG);
    printf("  Core_SP      : 0x%08X  (reg 0x%05X)\n", (unsigned)raw_sp, (unsigned)AIERT_CORE_SP_REG);
    /* Bit-field decode of Core_Status */
    printf("  [21] PROC_BUS_STALL  : %s\n", (raw_status & 0x00200000u) ? "YES" : "no");
    printf("  [20] CORE_DONE       : %s\n", (raw_status & 0x00100000u) ? "YES" : "no");
    printf("  [19] ERROR_HALT      : %s\n", (raw_status & 0x00080000u) ? "YES" : "no");
    printf("  [18] ECC_SCRUB_STALL : %s\n", (raw_status & 0x00040000u) ? "YES" : "no");
    printf("  [17] ECC_ERR_STALL   : %s\n", (raw_status & 0x00020000u) ? "YES" : "no");
    printf("  [16] DEBUG_HALT      : %s\n", (raw_status & 0x00010000u) ? "YES" : "no");
    printf("  [15] CASCADE_STALL_MCD: %s\n", (raw_status & 0x00008000u) ? "YES" : "no");
    printf("  [14] CASCADE_STALL_SCD: %s\n", (raw_status & 0x00004000u) ? "YES" : "no");
    printf("  [12] STREAM_STALL_MS0: %s\n", (raw_status & 0x00001000u) ? "YES" : "no");
    printf("  [10] STREAM_STALL_SS0: %s\n", (raw_status & 0x00000400u) ? "YES" : "no");
    printf("  [ 9] LOCK_STALL_E    : %s\n", (raw_status & 0x00000200u) ? "YES" : "no");
    printf("  [ 8] LOCK_STALL_N    : %s\n", (raw_status & 0x00000100u) ? "YES" : "no");
    printf("  [ 7] LOCK_STALL_W    : %s\n", (raw_status & 0x00000080u) ? "YES" : "no");
    printf("  [ 6] LOCK_STALL_S    : %s\n", (raw_status & 0x00000040u) ? "YES" : "no");
    printf("  [ 5] MEMORY_STALL_E  : %s\n", (raw_status & 0x00000020u) ? "YES" : "no");
    printf("  [ 4] MEMORY_STALL_N  : %s\n", (raw_status & 0x00000010u) ? "YES" : "no");
    printf("  [ 3] MEMORY_STALL_W  : %s\n", (raw_status & 0x00000008u) ? "YES" : "no");
    printf("  [ 2] MEMORY_STALL_S  : %s\n", (raw_status & 0x00000004u) ? "YES" : "no");
    printf("  [ 1] RESET           : %s\n", (raw_status & 0x00000002u) ? "YES" : "no");
    printf("  [ 0] ENABLE          : %s\n", (raw_status & 0x00000001u) ? "YES" : "no");
    /* LR from XAie API (not directly in Core_Status register) */
    if (!in_reset)
        printf("  LR             : 0x%08X\n", (unsigned)lr);

    /* Read core module event status register 0 for a live activity snapshot.
     * REG0 (0x00034200): event IDs  0-31 — contains stall / active bits (AIE2PS) */
    u64 evt_reg0_off = ((u64)tile.Col << 25) | ((u64)tile.Row << 20) | AIERT_CORE_CORE_EVT_STATUS_REG;
    u32 evt0 = 0;
    AieRC rc_e0 = XAie_Read32(dev, evt_reg0_off, &evt0);
    if (rc_e0 == XAIE_OK) {
        printf("  CoreEvtReg0  : 0x%08X  (event IDs  0-31)\n", (unsigned)evt0);
        int active = (int)((evt0 >> AIERT_CORE_EVT0_ACTIVE_BIT) & 1u);
        int disabled = (int)((evt0 >> AIERT_CORE_EVT0_DISABLED_BIT) & 1u);
        int ms_stall = (int)((evt0 >> AIERT_CORE_EVT0_MS_STALL_BIT) & 1u);
        int stream_stall = (int)((evt0 >> AIERT_CORE_EVT0_STREAM_STALL_BIT) & 1u);
        int cascade_stall = (int)((evt0 >> AIERT_CORE_EVT0_CASCADE_STALL_BIT) & 1u);
        int lock_stall = (int)((evt0 >> AIERT_CORE_EVT0_LOCK_STALL_BIT) & 1u);
        int ecc_err_stall = (int)((evt0 >> AIERT_CORE_EVT0_ECC_ERR_STALL_BIT) & 1u);
        int ecc_scrub = (int)((evt0 >> AIERT_CORE_EVT0_ECC_SCRUB_STALL_BIT) & 1u);
        printf("  ACTIVE       : %s\n", active ? "YES" : "no");
        printf("  DISABLED     : %s\n", disabled ? "YES" : "no");
        printf("  MS_STALL     : %s\n", ms_stall ? "*** STALL ***" : "no");
        printf("  STREAM_STALL : %s\n", stream_stall ? "*** STALL ***" : "no");
        printf("  CASCADE_STALL: %s\n", cascade_stall ? "*** STALL ***" : "no");
        printf("  LOCK_STALL   : %s\n", lock_stall ? "*** STALL ***" : "no");
        printf("  ECC_ERR_STALL: %s\n", ecc_err_stall ? "*** STALL ***" : "no");
        printf("  ECC_SCRUB    : %s\n", ecc_scrub ? "*** STALL ***" : "no");
        if (disabled)
            printf("  *** WARN: core is DISABLED\n");
        if (ms_stall || stream_stall || cascade_stall || lock_stall || ecc_err_stall || ecc_scrub)
            printf("  *** WARN: core has active stall(s)\n");
    } else {
        printf("  CoreEvtReg   : read failed (rc0=%d)\n", (int)rc_e0);
    }
}

void AieRt_PrintCoreStatusAll(XAie_DevInst *dev, const XAie_LocType *tiles, uint32_t num_tiles) {
    printf("[AieRt_Debug] ===== Core Status (%u tiles) =====\n", (unsigned)num_tiles);
    for (uint32_t i = 0; i < num_tiles; i++) {
        if (s_is_core(tiles[i]))
            AieRt_PrintCoreStatus(dev, tiles[i]);
    }
}

/* --------------------------------------------------------------------------
 * Core module activity / stall event analysis
 *
 * Uses AIERT_CORE_CORE_EVT_STATUS_REG (0x00034200) for event IDs 0-31.
 *
 * Relevant events (AIE2PS Table 4-8, all in register 0):
 *   bit 23  ID 23  MEMORY_STALL
 *   bit 24  ID 24  STREAM_STALL
 *   bit 25  ID 25  CASCADE_STALL
 *   bit 26  ID 26  LOCK_STALL
 *   bit 27  ID 27  DEBUG_HALTED
 *   bit 28  ID 28  ACTIVE
 *   bit 29  ID 29  DISABLED
 *   bit 30  ID 30  ECC_ERROR_STALL
 *   bit 31  ID 31  ECC_SCRUBBING_STALL
 * -------------------------------------------------------------------------- */

void AieRt_PrintCoreActivityEvents(XAie_DevInst *dev, XAie_LocType tile) {
    if (!s_is_core(tile)) {
        printf("[AieRt_Debug] tile(%u,%u): not a core tile, skip core activity events\n", (unsigned)tile.Col,
               (unsigned)tile.Row);
        return;
    }

    /* Read core module event status register 0 (event IDs 0-31, contains stall/active bits) */
    u64 reg0_off = ((u64)tile.Col << 25) | ((u64)tile.Row << 20) | AIERT_CORE_CORE_EVT_STATUS_REG;
    u32 evt0 = 0;
    AieRC rc = XAie_Read32(dev, reg0_off, &evt0);
    if (rc != XAIE_OK) {
        printf("[AieRt_Debug] tile(%u,%u): failed to read core_evt_status0 rc=%d\n", (unsigned)tile.Col,
               (unsigned)tile.Row, (int)rc);
        return;
    }

    int active = (int)((evt0 >> AIERT_CORE_EVT0_ACTIVE_BIT) & 1u);
    int disabled = (int)((evt0 >> AIERT_CORE_EVT0_DISABLED_BIT) & 1u);
    int ms_stall = (int)((evt0 >> AIERT_CORE_EVT0_MS_STALL_BIT) & 1u);
    int stream_stall = (int)((evt0 >> AIERT_CORE_EVT0_STREAM_STALL_BIT) & 1u);
    int cascade_stall = (int)((evt0 >> AIERT_CORE_EVT0_CASCADE_STALL_BIT) & 1u);
    int lock_stall = (int)((evt0 >> AIERT_CORE_EVT0_LOCK_STALL_BIT) & 1u);
    int ecc_err_stall = (int)((evt0 >> AIERT_CORE_EVT0_ECC_ERR_STALL_BIT) & 1u);
    int ecc_scrub = (int)((evt0 >> AIERT_CORE_EVT0_ECC_SCRUB_STALL_BIT) & 1u);

    printf("[AieRt_Debug] Core tile(%u,%u) core_evt_status0(0x%llX)=0x%08X\n", (unsigned)tile.Col, (unsigned)tile.Row,
           (unsigned long long)reg0_off, (unsigned)evt0);

    /* Activity state */
    printf("  ACTIVE          : %s\n", active ? "YES" : "no");
    printf("  DISABLED        : %s\n", disabled ? "YES" : "no");

    /* Stall events */
    printf("  MEMORY_STALL    : %s\n", ms_stall ? "*** STALL ***" : "no");
    printf("  STREAM_STALL    : %s\n", stream_stall ? "*** STALL ***" : "no");
    printf("  CASCADE_STALL   : %s\n", cascade_stall ? "*** STALL ***" : "no");
    printf("  LOCK_STALL      : %s\n", lock_stall ? "*** STALL ***" : "no");
    printf("  ECC_ERR_STALL   : %s\n", ecc_err_stall ? "*** STALL ***" : "no");
    printf("  ECC_SCRUB_STALL : %s\n", ecc_scrub ? "*** STALL ***" : "no");

    /* Summary warnings */
    if (disabled)
        printf("  *** WARN: core is DISABLED\n");
    if (ms_stall || stream_stall || cascade_stall || lock_stall || ecc_err_stall || ecc_scrub)
        printf("  *** WARN: core has active stall(s) - check stall lines above\n");
}

void AieRt_PrintCoreActivityEventsAll(XAie_DevInst *dev, const XAie_LocType *tiles, uint32_t num_tiles) {
    printf("[AieRt_Debug] ===== Core Activity Events (%u tiles) =====\n", (unsigned)num_tiles);
    for (uint32_t i = 0; i < num_tiles; i++) {
        if (s_is_core(tiles[i]))
            AieRt_PrintCoreActivityEvents(dev, tiles[i]);
    }
}

/* --------------------------------------------------------------------------
 * Core tile DMA BD event analysis
 *
 * The core tile Memory Module event status register 0 at tile-relative offset
 * 0x00014200 holds sticky bits for hardware-local event IDs 0-31.
 * (Full address: 0x20000000000 + (col<<25) + (row<<20) + 0x00014200)
 *
 * From debug.md: DMA_S2MM_0_START_TASK = local event 19 = bit 19.
 * Consecutive layout for all four DMA channels:
 *   bit 19  DMA_S2MM_0_START_TASK       bit 23  DMA_S2MM_0_FINISHED_TASK
 *   bit 20  DMA_S2MM_1_START_TASK       bit 24  DMA_S2MM_1_FINISHED_TASK
 *   bit 21  DMA_MM2S_0_START_TASK       bit 25  DMA_MM2S_0_FINISHED_TASK
 *   bit 22  DMA_MM2S_1_START_TASK       bit 26  DMA_MM2S_1_FINISHED_TASK
 *
 * A channel that has a START bit set but no FINISH bit indicates a stall or
 * incomplete transfer at the moment the snapshot was taken.
 * -------------------------------------------------------------------------- */

void AieRt_PrintCoreTileDmaBdEvents(XAie_DevInst *dev, XAie_LocType tile) {
    if (!s_is_core(tile)) {
        printf("[AieRt_Debug] tile(%u,%u): not a core tile, skip BD events\n", (unsigned)tile.Col, (unsigned)tile.Row);
        return;
    }

    /* Read memory module event status register directly */
    u64 reg_off = ((u64)tile.Col << 25) | ((u64)tile.Row << 20) | AIERT_CORE_MEM_EVT_STATUS_REG;
    u32 evt = 0;
    AieRC rc_ev = XAie_Read32(dev, reg_off, &evt);
    if (rc_ev != XAIE_OK) {
        printf("[AieRt_Debug] tile(%u,%u): failed to read mem_evt_status rc=%d\n", (unsigned)tile.Col,
               (unsigned)tile.Row, (int)rc_ev);
        return;
    }

    int s2mm0_start = (int)((evt >> AIERT_EVT_DMA_S2MM_0_START_BIT) & 1u);
    int s2mm1_start = (int)((evt >> AIERT_EVT_DMA_S2MM_1_START_BIT) & 1u);
    int mm2s0_start = (int)((evt >> AIERT_EVT_DMA_MM2S_0_START_BIT) & 1u);
    int mm2s1_start = (int)((evt >> AIERT_EVT_DMA_MM2S_1_START_BIT) & 1u);
    int s2mm0_finish = (int)((evt >> AIERT_EVT_DMA_S2MM_0_FINISH_BIT) & 1u);
    int s2mm1_finish = (int)((evt >> AIERT_EVT_DMA_S2MM_1_FINISH_BIT) & 1u);
    int mm2s0_finish = (int)((evt >> AIERT_EVT_DMA_MM2S_0_FINISH_BIT) & 1u);
    int mm2s1_finish = (int)((evt >> AIERT_EVT_DMA_MM2S_1_FINISH_BIT) & 1u);

    printf("[AieRt_Debug] Core tile(%u,%u) DMA BD events  mem_evt_status(0x%08X)=0x%08X\n", (unsigned)tile.Col,
           (unsigned)tile.Row, reg_off, (unsigned)evt);

    /* START events */
    printf("  S2MM ch0  START_TASK   : %s\n", s2mm0_start ? "FIRED" : "-");
    printf("  S2MM ch1  START_TASK   : %s\n", s2mm1_start ? "FIRED" : "-");
    printf("  MM2S ch0  START_TASK   : %s\n", mm2s0_start ? "FIRED" : "-");
    printf("  MM2S ch1  START_TASK   : %s\n", mm2s1_start ? "FIRED" : "-");

    /* FINISHED events */
    printf("  S2MM ch0  FINISHED_TASK: %s\n", s2mm0_finish ? "FIRED" : "-");
    printf("  S2MM ch1  FINISHED_TASK: %s\n", s2mm1_finish ? "FIRED" : "-");
    printf("  MM2S ch0  FINISHED_TASK: %s\n", mm2s0_finish ? "FIRED" : "-");
    printf("  MM2S ch1  FINISHED_TASK: %s\n", mm2s1_finish ? "FIRED" : "-");

    /* Stall diagnosis: started but never finished */
    if (s2mm0_start && !s2mm0_finish)
        printf("  *** WARN: S2MM ch0 started but not finished (stall or in-flight?)\n");
    if (s2mm1_start && !s2mm1_finish)
        printf("  *** WARN: S2MM ch1 started but not finished (stall or in-flight?)\n");
    if (mm2s0_start && !mm2s0_finish)
        printf("  *** WARN: MM2S ch0 started but not finished (stall or in-flight?)\n");
    if (mm2s1_start && !mm2s1_finish)
        printf("  *** WARN: MM2S ch1 started but not finished (stall or in-flight?)\n");

    if (!s2mm0_start && !s2mm1_start && !mm2s0_start && !mm2s1_start)
        printf("  (no DMA activity recorded in this tile)\n");
}

void AieRt_PrintCoreTileDmaBdEventsAll(XAie_DevInst *dev, const XAie_LocType *tiles, uint32_t num_tiles) {
    printf("[AieRt_Debug] ===== Core Tile DMA BD Events (%u tiles) =====\n", (unsigned)num_tiles);
    for (uint32_t i = 0; i < num_tiles; i++) {
        if (s_is_core(tiles[i]))
            AieRt_PrintCoreTileDmaBdEvents(dev, tiles[i]);
    }
}

/* --------------------------------------------------------------------------
 * DMA BD info
 * -------------------------------------------------------------------------- */

void AieRt_PrintBdInfo(XAie_DevInst *dev, XAie_LocType tile, uint8_t bd_id) {
    XAie_DmaDesc desc;
    memset(&desc, 0, sizeof(desc));

    AieRC rc = XAie_DmaReadBd(dev, &desc, tile, (u16)bd_id);
    if (rc != XAIE_OK) {
        printf("[AieRt_Debug]   BD%-2u: read failed rc=%d\n", (unsigned)bd_id, (int)rc);
        return;
    }

    /* Extract fields from the descriptor struct.
     * XAie_DmaDesc has a union inside (AieMLDma or AieDma); we access
     * the generic fields that are present in both via the AddrDesc. */
    u64 addr = desc.AddrDesc.Address * 4; // the value is in 32bits words
    u32 len = desc.AddrDesc.Length * 4;   // the value is in 32bits words

    /* Lock fields */
    int acq_lock_id = (int)desc.LockDesc.LockAcqId;
    int acq_lock_val = (int)desc.LockDesc.LockAcqVal;
    int rel_lock_id = (int)desc.LockDesc.LockRelId;
    int rel_lock_val = (int)desc.LockDesc.LockRelVal;
    int acq_en = (int)desc.LockDesc.LockAcqEn;
    int rel_en = (int)desc.LockDesc.LockRelEn;

    /* Next BD */
    int next_bd_en = (int)desc.BdEnDesc.UseNxtBd;
    int next_bd_id = (int)desc.BdEnDesc.NxtBd;

    /* Packet */
    int pkt_en = (int)desc.PktDesc.PktEn;
    int pkt_id = (int)desc.PktDesc.PktId;
    int pkt_type = (int)desc.PktDesc.PktType;

    /* Out-of-order BD ID */
    int ooo_bd_id = (int)desc.BdEnDesc.OutofOrderBdId;
    int ooo_en = (int)desc.EnOutofOrderBdId;

    printf("[AieRt_Debug]   BD%-2u: addr=0x%llx len=%u bytes (tile=c-%u,r-%u)", (unsigned)bd_id,
           (unsigned long long)addr, (unsigned)len, (unsigned)tile.Col, (unsigned)tile.Row);

    if (next_bd_en)
        printf("  next->BD%u", (unsigned)next_bd_id);
    else
        printf("  next=(none)");

    if (pkt_en)
        printf("  pkt_en=%d pkt_id=%u pkt_type=%u", pkt_en, (unsigned)pkt_id, (unsigned)pkt_type);

    if (ooo_en)
        printf("  ooo_bd_id=%u", (unsigned)ooo_bd_id);

    printf("\n");
    printf("         lock acq: en=%d id=%d val=%d  rel: en=%d id=%d val=%d\n", acq_en, acq_lock_id, acq_lock_val,
           rel_en, rel_lock_id, rel_lock_val);

    /* Multi-dimensional addressing: stride/wrap for D0-D3 and iteration
     * XAie_DmaReadBd populates desc.MultiDimDesc from hardware registers.
     * Access via AieMlMultiDimDesc for AIEML/AIE2 devices. */
    {
        XAie_AieMlMultiDimDesc *md = &desc.MultiDimDesc.AieMlMultiDimDesc;
        int has_dim = 0;
        for (int d = 0; d < 4; d++) {
            if (md->DimDesc[d].StepSize != 0 || md->DimDesc[d].Wrap != 0)
                has_dim = 1;
        }
        if (md->IterDesc.StepSize != 0 || md->IterDesc.Wrap != 0)
            has_dim = 1;

        if (has_dim) {
            printf("         dims:");
            for (int d = 0; d < 4; d++) {
                if (md->DimDesc[d].StepSize != 0 || md->DimDesc[d].Wrap != 0)
                    printf("  D%d(step=%u,wrap=%u)", d, (unsigned)md->DimDesc[d].StepSize,
                           (unsigned)md->DimDesc[d].Wrap);
            }
            if (md->IterDesc.StepSize != 0 || md->IterDesc.Wrap != 0)
                printf("  Iter(step=%u,wrap=%u,curr=%u)", (unsigned)md->IterDesc.StepSize, (unsigned)md->IterDesc.Wrap,
                       (unsigned)md->IterCurr);
            printf("\n");
        }
    }
}

void AieRt_PrintAllBds(XAie_DevInst *dev, XAie_LocType tile) {
    u8 num_bds = 0;
    AieRC rc = XAie_DmaGetNumBds(dev, tile, &num_bds);
    if (rc != XAIE_OK || num_bds == 0) {
        printf("[AieRt_Debug] tile(%u,%u): cannot determine BD count (rc=%d)\n", (unsigned)tile.Col, (unsigned)tile.Row,
               (int)rc);
        return;
    }
    printf("[AieRt_Debug] tile(%u,%u) BDs (total=%u, printing configured only):\n", (unsigned)tile.Col,
           (unsigned)tile.Row, (unsigned)num_bds);
    int printed = 0;
    for (u8 i = 0; i < num_bds; i++) {
        /* Quick check: read the BD via API; skip if addr and len are both zero */
        XAie_DmaDesc desc;
        memset(&desc, 0, sizeof(desc));
        AieRC rbd = XAie_DmaReadBd(dev, &desc, tile, (u16)i);
        if (rbd != XAIE_OK)
            continue;
        if (desc.AddrDesc.Address == 0 && desc.AddrDesc.Length == 0)
            continue;
        AieRt_PrintBdInfo(dev, tile, i);
        printed++;
    }
    if (printed == 0)
        printf("[AieRt_Debug]   (no configured BDs)\n");
}

/* --------------------------------------------------------------------------
 * Raw BD register dump for shim tiles
 *
 * Reads all 16 BD slots via XAie_Read32 at tile-relative offsets.
 * Decodes stride/wrap/iteration/lock/packet fields from raw register words.
 * -------------------------------------------------------------------------- */

void AieRt_PrintShimBdRawAll(XAie_DevInst *dev, uint8_t col) {
    /* Select BD layout based on AIE generation */
    uint32_t bd_base, bd_stride, bd_words;
    const char *gen_tag;
    if (g_aiert_gen == AIERT_GEN_AIE2PS) {
        bd_base = AIERT_SHIM_BD_BASE_2PS;
        bd_stride = AIERT_SHIM_BD_STRIDE_2PS;
        bd_words = AIERT_SHIM_BD_WORDS_2PS;
        gen_tag = "AIE2PS";
    } else {
        bd_base = AIERT_SHIM_BD_BASE_ML;
        bd_stride = AIERT_SHIM_BD_STRIDE_ML;
        bd_words = AIERT_SHIM_BD_WORDS_ML;
        gen_tag = "AIEML";
    }
    printf("[AieRt_Debug] ===== Shim tile(%u,0) Raw BD Dump [%s] (16 BDs, base=0x%x stride=0x%x words=%u) =====\n",
           (unsigned)col, gen_tag, (unsigned)bd_base, (unsigned)bd_stride, (unsigned)bd_words);

    int printed = 0;
    u32 w[9]; /* max 9 words (AIE2PS) */
    for (uint32_t bd = 0; bd < AIERT_SHIM_BD_COUNT; bd++) {
        uint32_t base = bd_base + bd * bd_stride;

        /* Read all words of this BD.
         * Shim tile address: (col << 25) | (row << 20) | reg_offset, row=0. */
        u64 tile_base = (u64)col << 25; /* shim row=0 */
        for (uint32_t wi = 0; wi < bd_words; wi++) {
            AieRC rc = XAie_Read32(dev, tile_base | (u64)(base + wi * 4), &w[wi]);
            if (rc != XAIE_OK)
                w[wi] = 0;
        }

        /* Always print the first 8 BDs regardless of content for debug
         * visibility (post-completion/error state may have zeroed BDs).
         * Skip remaining BDs only if Buffer_Length (word0) is zero. */
        if (bd >= 8 && w[0] == 0)
            continue;

        /* Decode word 0: Buffer_Length */
        uint32_t buf_len = w[0];

        /* Decode word 1: Base_Address_Low */
        uint32_t addr_lo = w[1];

        /* Decode word 2: Enable_Packet[30], Out_of_Order_BD_ID[29:24],
         * Packet_ID[23:19], Packet_Type[18:16], Base_Address_High[15:0] */
        uint32_t addr_hi = w[2] & 0xFFFFu;
        uint32_t enable_pkt = (w[2] >> 30) & 0x1u;
        uint32_t ooo_bd_id = (w[2] >> 24) & 0x3Fu;
        uint32_t pkt_id = (w[2] >> 19) & 0x1Fu;
        uint32_t pkt_type = (w[2] >> 16) & 0x7u;

        /* Decode word 3: D0_Stepsize[19:0], D0_Wrap[29:20] */
        uint32_t d0_step = w[3] & 0xFFFFFu;
        uint32_t d0_wrap = (w[3] >> 20) & 0x3FFu;

        /* Decode word 4: D1_Stepsize[19:0], D1_Wrap[29:20] */
        uint32_t d1_step = w[4] & 0xFFFFFu;
        uint32_t d1_wrap = (w[4] >> 20) & 0x3FFu;

        /* Decode word 5: D2_Stepsize[19:0], D2_Wrap[29:20] */
        uint32_t d2_step = w[5] & 0xFFFFFu;
        uint32_t d2_wrap = (w[5] >> 20) & 0x3FFu;

        /* Decode word 6: Iter_Stepsize[19:0], Iter_Wrap[25:20], Iter_Current[31:26] */
        uint32_t iter_step = w[6] & 0xFFFFFu;
        uint32_t iter_wrap = (w[6] >> 20) & 0x3Fu;
        uint32_t iter_curr = (w[6] >> 26) & 0x3Fu;

        /* Word 7 is the lock/valid word for BOTH AIEML and AIE2PS shim NOC BDs.
         * (AIE2PS word 8 at offset +0x20 holds BASE_ADDRESS_56_48 extra address bits.)
         * Bit layout (both gens — from xaie2psgbl_params.h / xaiemlgbl_params.h):
         *   [3:0]   LOCK_ACQ_ID
         *   [11:5]  LOCK_ACQ_VALUE (7-bit signed)
         *   [12]    LOCK_ACQ_ENABLE
         *   [16:13] LOCK_REL_ID
         *   [24:18] LOCK_REL_VALUE (7-bit signed)
         *   [25]    VALID_BD
         *   [26]    USE_NEXT_BD
         *   [30:27] NEXT_BD
         *   [31]    TLAST_SUPPRESS  */
        uint32_t lock_word = w[7];
        uint32_t acq_id = lock_word & 0xFu;
        int32_t acq_val = (int32_t)((int8_t)(((lock_word >> 5) & 0x7Fu) << 1) >> 1);
        uint32_t acq_en = (lock_word >> 12) & 0x1u;
        uint32_t rel_id = (lock_word >> 13) & 0xFu;
        int32_t rel_val = (int32_t)((int8_t)(((lock_word >> 18) & 0x7Fu) << 1) >> 1);
        uint32_t valid = (lock_word >> 25) & 0x1u;
        uint32_t use_next = (lock_word >> 26) & 0x1u;
        uint32_t next_bd = (lock_word >> 27) & 0xFu;
        uint32_t rel_en = 0; /* LOCK_REL_ENABLE is not in shim BD word7 — infer from rel_id!=0 */
        (void)rel_en;

        /* Print header with raw words */
        if (bd_words == 9)
            printf("[AieRt_Debug]   BD%-2u raw: [%08x %08x %08x %08x %08x %08x %08x %08x %08x]\n", (unsigned)bd, w[0],
                   w[1], w[2], w[3], w[4], w[5], w[6], w[7], w[8]);
        else
            printf("[AieRt_Debug]   BD%-2u raw: [%08x %08x %08x %08x %08x %08x %08x %08x]\n", (unsigned)bd, w[0], w[1],
                   w[2], w[3], w[4], w[5], w[6], w[7]);

        /* Print decoded fields */
        uint64_t full_addr = ((uint64_t)addr_hi << 32) | (uint64_t)addr_lo;
        printf("[AieRt_Debug]     addr=0x%llx len=%u valid=%u", (unsigned long long)full_addr, buf_len, valid);
        if (use_next)
            printf("  next->BD%u", next_bd);
        else
            printf("  next=(none)");
        printf("  enable_pkt=%u pkt_id=%u pkt_type=%u ooo_bd_id=%u", enable_pkt, pkt_id, pkt_type, ooo_bd_id);
        printf("\n");

        /* Print dimension strides/wraps */
        printf("[AieRt_Debug]     D0: step=%u wrap=%u  D1: step=%u wrap=%u  D2: step=%u wrap=%u\n", d0_step, d0_wrap,
               d1_step, d1_wrap, d2_step, d2_wrap);

        /* Print iteration */
        if (iter_step || iter_wrap || iter_curr)
            printf("[AieRt_Debug]     Iter: step=%u wrap=%u current=%u\n", iter_step, iter_wrap, iter_curr);

        /* Print lock */
        if (acq_en || rel_en)
            printf("[AieRt_Debug]     Lock: acq_en=%u id=%u val=%d  rel_en=%u id=%u val=%d\n", acq_en, acq_id, acq_val,
                   rel_en, rel_id, rel_val);

        printed++;
    }

    printf("[AieRt_Debug]   (%d BDs printed, first 8 always shown)\n", printed);
    printf("[AieRt_Debug] ===== End Shim tile(%u,0) Raw BD Dump =====\n", (unsigned)col);
}

/* --------------------------------------------------------------------------
 * DMA channel status
 * -------------------------------------------------------------------------- */

AieRC AieRt_DmaGetChannelStatusFull(XAie_DevInst *dev, XAie_LocType tile, uint8_t ch, XAie_DmaDirection dir,
                                    uint32_t *status) {
    if (dev == NULL || status == NULL)
        return XAIE_INVALID_ARGS;
    if (ch > 1u)
        return XAIE_INVALID_ARGS;

    /* Select S2MM / MM2S base offset depending on tile type.
     * Core and MemTile offsets are identical across AIEML and AIE2PS.
     * Shim NOC tile offsets differ: AIEML vs AIE2PS. */
    u32 s2mm_base, mm2s_base;
    if (s_is_shim(tile)) {
        if (g_aiert_gen == AIERT_GEN_AIE2PS) {
            s2mm_base = AIERT_DMA_SHIM_S2MM_STATUS_BASE_2PS;
            mm2s_base = AIERT_DMA_SHIM_MM2S_STATUS_BASE_2PS;
        } else {
            s2mm_base = AIERT_DMA_SHIM_S2MM_STATUS_BASE_ML;
            mm2s_base = AIERT_DMA_SHIM_MM2S_STATUS_BASE_ML;
        }
    } else if (s_is_memtile(tile)) {
        s2mm_base = AIERT_DMA_MEM_S2MM_STATUS_BASE;
        mm2s_base = AIERT_DMA_MEM_MM2S_STATUS_BASE;
    } else if (s_is_core(tile)) {
        s2mm_base = AIERT_DMA_CORE_S2MM_STATUS_BASE;
        mm2s_base = AIERT_DMA_CORE_MM2S_STATUS_BASE;
    } else {
        return XAIE_INVALID_TILE;
    }

    u32 reg_offset = (dir == DMA_S2MM) ? s2mm_base : mm2s_base;
    reg_offset += ch * AIERT_DMA_CHNUM_OFFSET;

    u64 addr = ((u64)tile.Col << 25) | ((u64)tile.Row << 20) | (u64)reg_offset;
    return XAie_Read32(dev, addr, status);
}

AieRt_DmaChStatusDecoded AieRt_DecodeDmaChStatus(uint32_t raw) {
    AieRt_DmaChStatusDecoded d;
    memset(&d, 0, sizeof(d));
    d.raw = raw;
    d.status = (uint8_t)((raw & AIERT_DMA_STATUS_MASK));
    d.stall_lock_acq = (raw & AIERT_DMA_STALL_LOCK_ACQ_MASK) ? 1u : 0u;
    d.stall_lock_rel = (raw & AIERT_DMA_STALL_LOCK_REL_MASK) ? 1u : 0u;
    d.stall_stream = (raw & AIERT_DMA_STALL_STREAM_MASK) ? 1u : 0u;
    d.stall_tct = (raw & AIERT_DMA_STALL_TCT_MASK) ? 1u : 0u;
    d.err_bd_unavail = (raw & AIERT_DMA_ERR_BD_UNAVAIL_MASK) ? 1u : 0u;
    d.err_bd_invalid = (raw & AIERT_DMA_ERR_BD_INVALID_MASK) ? 1u : 0u;
    d.err_fot_len = (raw & AIERT_DMA_ERR_FOT_LEN_MASK) ? 1u : 0u;
    d.err_fot_bds = (raw & AIERT_DMA_ERR_FOT_BDS_MASK) ? 1u : 0u;
    d.err_axi_decode = (raw & AIERT_DMA_ERR_AXI_DECODE_MASK) ? 1u : 0u;
    d.err_axi_slave = (raw & AIERT_DMA_ERR_AXI_SLAVE_MASK) ? 1u : 0u;
    d.task_q_overflow = (raw & AIERT_DMA_TASK_Q_OVERFLOW_MASK) ? 1u : 0u;
    d.channel_running = (raw & AIERT_DMA_CHANNEL_RUNNING_MASK) ? 1u : 0u;
    d.task_q_size = (uint8_t)((raw & AIERT_DMA_TASK_Q_SIZE_MASK) >> AIERT_DMA_TASK_Q_SIZE_LSB);
    d.cur_bd = (uint8_t)((raw & AIERT_DMA_CUR_BD_MASK) >> AIERT_DMA_CUR_BD_LSB);
    return d;
}

void AieRt_PrintDmaChStatus(XAie_DevInst *dev, XAie_LocType tile, uint8_t ch, XAie_DmaDirection dir) {
    u32 raw = 0;
    AieRC rc = AieRt_DmaGetChannelStatusFull(dev, tile, ch, dir, &raw);
    if (rc != XAIE_OK) {
        printf("[AieRt_Debug]   tile(%u,%u) ch%u %s: GetChannelStatus failed rc=%d\n", (unsigned)tile.Col,
               (unsigned)tile.Row, (unsigned)ch, s_dir_str(dir), (int)rc);
        return;
    }

    AieRt_DmaChStatusDecoded d = AieRt_DecodeDmaChStatus(raw);

    printf("[AieRt_Debug]   tile(%u,%u) ch%u %s: raw=0x%08X  %s  running=%d  q_size=%u  cur_bd=%u\n",
           (unsigned)tile.Col, (unsigned)tile.Row, (unsigned)ch, s_dir_str(dir), (unsigned)raw, s_status_str(d.status),
           (int)d.channel_running, (unsigned)d.task_q_size, (unsigned)d.cur_bd);

    /* Report any stalls */
    if (d.stall_lock_acq || d.stall_lock_rel || d.stall_stream || d.stall_tct) {
        printf("             STALLED: lock_acq=%d lock_rel=%d stream=%d tct=%d\n", (int)d.stall_lock_acq,
               (int)d.stall_lock_rel, (int)d.stall_stream, (int)d.stall_tct);
    }

    /* Report any errors */
    if (d.err_bd_unavail || d.err_bd_invalid || d.err_fot_len || d.err_fot_bds || d.err_axi_decode || d.err_axi_slave ||
        d.task_q_overflow) {
        printf("             ERRORS: bd_unavail=%d bd_invalid=%d fot_len=%d fot_bds=%d "
               "axi_decode=%d axi_slave=%d q_overflow=%d\n",
               (int)d.err_bd_unavail, (int)d.err_bd_invalid, (int)d.err_fot_len, (int)d.err_fot_bds,
               (int)d.err_axi_decode, (int)d.err_axi_slave, (int)d.task_q_overflow);
    }

    /* If the channel is actively processing, show the current BD contents */
    if (d.channel_running || d.status == AIERT_DMA_STATUS_RUNNING) {
        printf("             Current BD contents:\n");
        AieRt_PrintBdInfo(dev, tile, d.cur_bd);
    }
}

void AieRt_PrintShimDmaStatus(XAie_DevInst *dev, uint8_t col) {
    XAie_LocType shim = XAie_TileLoc(col, XAIE_SHIM_ROW);
    printf("[AieRt_Debug] Shim tile(%u,0) DMA status:\n", (unsigned)col);
    AieRt_PrintDmaChStatus(dev, shim, 0, DMA_S2MM);
    AieRt_PrintDmaChStatus(dev, shim, 1, DMA_S2MM);
    AieRt_PrintDmaChStatus(dev, shim, 0, DMA_MM2S);
    AieRt_PrintDmaChStatus(dev, shim, 1, DMA_MM2S);
}

void AieRt_PrintShimPlModuleEvents(XAie_DevInst *dev, uint8_t col) {
    u64 base = (u64)col << 25; /* shim row=0 */
    u32 reg[6] = {0, 0, 0, 0, 0, 0};
    XAie_Read32(dev, base | AIERT_SHIM_PL_EVT_STATUS_REG0, &reg[0]);
    XAie_Read32(dev, base | AIERT_SHIM_PL_EVT_STATUS_REG1, &reg[1]);
    XAie_Read32(dev, base | AIERT_SHIM_PL_EVT_STATUS_REG2, &reg[2]);
    XAie_Read32(dev, base | AIERT_SHIM_PL_EVT_STATUS_REG3, &reg[3]);
    XAie_Read32(dev, base | AIERT_SHIM_PL_EVT_STATUS_REG4, &reg[4]);
    XAie_Read32(dev, base | AIERT_SHIM_PL_EVT_STATUS_REG5, &reg[5]);

    const char *gen_str = (g_aiert_gen == AIERT_GEN_AIE2PS) ? "AIE2PS" : "AIEML";
    printf("[AieRt_Debug] Shim(%u,0) PL module events [gen=%s]"
           "  reg0=0x%08X  reg1=0x%08X  reg2=0x%08X  reg3=0x%08X"
           "  reg4=0x%08X  reg5=0x%08X\n",
           (unsigned)col, gen_str, (unsigned)reg[0], (unsigned)reg[1], (unsigned)reg[2], (unsigned)reg[3],
           (unsigned)reg[4], (unsigned)reg[5]);

    int any = 0;
    for (uint32_t r = 0; r < 6u; r++) {
        for (int bit = 0; bit < 32; bit++) {
            if (!((reg[r] >> bit) & 1u))
                continue;
            uint32_t id = r * 32u + (uint32_t)bit;
            if (id > AIERT_PL_EVT_MAX)
                continue;
            const char *name = AieRt_EventName(XAIE_PL_MOD, id);
            /* Flag ranges per aie2psshimevent.md Table 7-12:
             *   NoC 0 DMA activity       : IDs 14-37
             *   NoC 0 DMA stalled lock   : IDs 26-29
             *   NoC 0 DMA stream starv/bp: IDs 30-33
             *   NoC 0 DMA memory bp/starv: IDs 34-37
             *   NoC 1 DMA activity       : IDs 39-62
             *   NoC 1 DMA stalled lock   : IDs 51-54
             *   NoC 1 DMA stream starv/bp: IDs 55-58
             *   NoC 1 DMA memory bp/starv: IDs 59-62
             *   Errors                   : IDs 114-129
             */
            const char *flag = "";
            if ((id >= 14u && id <= 37u) || (id >= 39u && id <= 62u))
                flag = " [DMA]";
            if ((id >= 26u && id <= 29u) || (id >= 51u && id <= 54u))
                flag = " *** DMA STALLED_LOCK ***";
            if (id == 30u || id == 31u || id == 55u || id == 56u)
                flag = " *** S2MM STREAM STARVATION ***";
            if (id == 32u || id == 33u || id == 57u || id == 58u)
                flag = " *** MM2S STREAM BACKPRESSURE ***";
            if ((id >= 34u && id <= 35u) || (id >= 59u && id <= 60u))
                flag = " *** MEMORY BACKPRESSURE ***";
            if ((id >= 36u && id <= 37u) || (id >= 61u && id <= 62u))
                flag = " *** MEMORY STARVATION ***";
            if (id >= 113u && id <= 129u)
                flag = " *** ERROR ***";
            printf("  PL evt %3u  %s%s\n", (unsigned)id, name, flag);
            any = 1;
        }
    }
    if (!any)
        printf("  (no PL module events set)\n");
}

void AieRt_PrintCoreTileDmaStatus(XAie_DevInst *dev, XAie_LocType tile) {
    if (!s_is_core(tile)) {
        printf("[AieRt_Debug] tile(%u,%u) is not a core tile\n", (unsigned)tile.Col, (unsigned)tile.Row);
        return;
    }
    printf("[AieRt_Debug] Core tile(%u,%u) DMA status:\n", (unsigned)tile.Col, (unsigned)tile.Row);
    AieRt_PrintDmaChStatus(dev, tile, 0, DMA_S2MM);
    AieRt_PrintDmaChStatus(dev, tile, 1, DMA_S2MM);
    AieRt_PrintDmaChStatus(dev, tile, 0, DMA_MM2S);
    AieRt_PrintDmaChStatus(dev, tile, 1, DMA_MM2S);
}

/* --------------------------------------------------------------------------
 * Verification
 * -------------------------------------------------------------------------- */

int AieRt_VerifyIoDescriptors(XAie_DevInst *dev, const struct_io *ios, uint32_t num_ios) {
    int fail_count = 0;

    printf("[AieRt_Debug] ===== IO Descriptor Verification (%u IOs) =====\n", (unsigned)num_ios);

    for (uint32_t i = 0; i < num_ios; i++) {
        const struct_io *io = &ios[i];
        u32 raw = 0;
        AieRC rc = AieRt_DmaGetChannelStatusFull(dev, io->tile_loc, io->channel_id, io->direction, &raw);
        if (rc != XAIE_OK) {
            printf("[AieRt_Debug] IO[%u] tile(%u,%u) ch%u %s: GetChannelStatus FAILED rc=%d\n", (unsigned)i,
                   (unsigned)io->tile_loc.Col, (unsigned)io->tile_loc.Row, (unsigned)io->channel_id,
                   s_dir_str(io->direction), (int)rc);
            fail_count++;
            continue;
        }

        AieRt_DmaChStatusDecoded d = AieRt_DecodeDmaChStatus(raw);

        /* Error conditions: any error bit or stall that is not lock-acq transient */
        int has_error = (int)(d.err_bd_unavail | d.err_bd_invalid | d.err_fot_len | d.err_fot_bds | d.err_axi_decode |
                              d.err_axi_slave | d.task_q_overflow);
        int has_stall = (int)(d.stall_lock_acq | d.stall_lock_rel | d.stall_stream | d.stall_tct);

        /* A well-completed IO should be Idle or Running (task queue draining).
         * We flag it as FAIL only if there are hard errors. */
        int io_fail = has_error;

        printf("[AieRt_Debug] IO[%u] tile(%u,%u) ch%u %s bd_id=%u: status=%s running=%d q=%u cur_bd=%u  => %s\n",
               (unsigned)i, (unsigned)io->tile_loc.Col, (unsigned)io->tile_loc.Row, (unsigned)io->channel_id,
               s_dir_str(io->direction), (unsigned)io->bd_id, s_status_str(d.status), (int)d.channel_running,
               (unsigned)d.task_q_size, (unsigned)d.cur_bd, io_fail ? "FAIL" : (has_stall ? "WARN-stall" : "OK"));

        if (has_error) {
            printf("  ERRORS: bd_unavail=%d bd_invalid=%d fot_len=%d fot_bds=%d "
                   "axi_decode=%d axi_slave=%d q_overflow=%d\n",
                   (int)d.err_bd_unavail, (int)d.err_bd_invalid, (int)d.err_fot_len, (int)d.err_fot_bds,
                   (int)d.err_axi_decode, (int)d.err_axi_slave, (int)d.task_q_overflow);
            fail_count++;
        }
        if (has_stall) {
            printf("  STALLS: lock_acq=%d lock_rel=%d stream=%d tct=%d\n", (int)d.stall_lock_acq, (int)d.stall_lock_rel,
                   (int)d.stall_stream, (int)d.stall_tct);
        }

        /* Print the BD that was registered for this IO */
        printf("  Registered BD contents (bd_id=%u):\n", (unsigned)io->bd_id);
        AieRt_PrintBdInfo(dev, io->tile_loc, io->bd_id);

        /* If channel is running, also print the BD currently being processed */
        if (d.channel_running && d.cur_bd != io->bd_id) {
            printf("  Current BD in flight (cur_bd=%u):\n", (unsigned)d.cur_bd);
            AieRt_PrintBdInfo(dev, io->tile_loc, d.cur_bd);
        }
    }

    printf("[AieRt_Debug] ===== Verification result: %d / %u FAILED =====\n", fail_count, (unsigned)num_ios);
    return fail_count;
}

/* --------------------------------------------------------------------------
 * Full debug snapshot
 * -------------------------------------------------------------------------- */

void AieRt_DebugSnapshot(XAie_DevInst *dev, const struct_io *ios, uint32_t num_ios, const XAie_LocType *tiles,
                         uint32_t num_tiles) {
    const char *gen_str = (g_aiert_gen == AIERT_GEN_AIE2PS) ? "AIE2PS" : "AIEML";
    printf("\n[AieRt_Debug] ============================================================\n");
    printf("[AieRt_Debug]  DEBUG SNAPSHOT  [gen=%s]\n", gen_str);
    printf("[AieRt_Debug] ============================================================\n");

    /* 1. Core status */
    AieRt_PrintCoreStatusAll(dev, tiles, num_tiles);

    /* 2. Stream switch configuration (JSON) — most useful for routing debug */
    {
        XAie_LocType ss_tiles[512];
        uint32_t ss_count = 0;

        /* Collect all known tiles (core + shim) to determine column/row span */
        uint8_t min_col = 0xFF, max_col = 0;
        uint8_t max_row = 0;

        for (uint32_t i = 0; i < num_tiles; i++) {
            if (tiles[i].Col < min_col)
                min_col = tiles[i].Col;
            if (tiles[i].Col > max_col)
                max_col = tiles[i].Col;
            if (tiles[i].Row > max_row)
                max_row = tiles[i].Row;
        }
        for (uint32_t i = 0; i < num_ios; i++) {
            if (ios[i].tile_loc.Col < min_col)
                min_col = ios[i].tile_loc.Col;
            if (ios[i].tile_loc.Col > max_col)
                max_col = ios[i].tile_loc.Col;
            if (ios[i].tile_loc.Row > max_row)
                max_row = ios[i].tile_loc.Row;
        }
        if (min_col > max_col) {
            min_col = 0;
            max_col = 0;
        }

        /* Scan the full column×row rectangle so that pass-through routing
         * tiles (e.g. shim col 6 → mem-tile rows 1-2 → core row 3) are
         * included.  print_all=0 means tiles with no active ports produce
         * empty JSON and are harmlessly skipped by the visualiser.           */
        for (uint8_t col = min_col; col <= max_col && ss_count < 512; col++) {
            for (uint8_t row = 0; row <= max_row && ss_count < 512; row++) {
                ss_tiles[ss_count++] = XAie_TileLoc(col, row);
            }
        }

        if (AIE_DEBUG_LEVEL(g_runtime_debug_level) >= 1)
            AieRt_PrintStreamSwitchConfigAll(dev, ss_tiles, ss_count, /*print_all=*/0);
    }

    /* 3. Core module activity / stall event analysis (ACTIVE, DISABLED, stalls) */
    AieRt_PrintCoreActivityEventsAll(dev, tiles, num_tiles);

    /* 4. Core tile DMA BD start/finish event analysis */
    AieRt_PrintCoreTileDmaBdEventsAll(dev, tiles, num_tiles);

    /* 5. MEM module event status — all set event bits with names */
    AieRt_PrintMemModuleEventsAll(dev, tiles, num_tiles);

    /* 6. CORE module event status — all set event bits with names */
    AieRt_PrintCoreModuleEventsAll(dev, tiles, num_tiles);

    /* 7. Shim DMA status — collect unique shim columns from ios */
    printf("[AieRt_Debug] ===== Shim DMA Status =====\n");
    {
        uint8_t seen_cols[64];
        uint32_t seen_count = 0;
        for (uint32_t i = 0; i < num_ios; i++) {
            if (!s_is_shim(ios[i].tile_loc))
                continue;
            uint8_t col = ios[i].tile_loc.Col;
            int already = 0;
            for (uint32_t j = 0; j < seen_count; j++) {
                if (seen_cols[j] == col) {
                    already = 1;
                    break;
                }
            }
            if (!already && seen_count < 64) {
                seen_cols[seen_count++] = col;
                AieRt_PrintShimDmaStatus(dev, col);
                AieRt_PrintShimPlModuleEvents(dev, col);
            }
        }
        if (seen_count == 0)
            printf("[AieRt_Debug]  (no shim tiles in ios list)\n");
    }

    /* 8. Core tile DMA status */
    printf("[AieRt_Debug] ===== Core Tile DMA Status =====\n");
    for (uint32_t i = 0; i < num_tiles; i++) {
        if (s_is_core(tiles[i]))
            AieRt_PrintCoreTileDmaStatus(dev, tiles[i]);
    }

    /* 9. Performance counters (MM2S BD finished counts) */
    AieRt_PrintCoreTilePerfCountersAll(dev, tiles, num_tiles);

    /* 10. BD contents for all referenced tiles */
    printf("[AieRt_Debug] ===== BD Contents (all referenced tiles) =====\n");
    {
        /* Unique tile list derived from ios + tiles */
        XAie_LocType seen[128];
        uint32_t seen_count = 0;

        /* From ios */
        for (uint32_t i = 0; i < num_ios; i++) {
            XAie_LocType t = ios[i].tile_loc;
            int found = 0;
            for (uint32_t j = 0; j < seen_count; j++) {
                if (seen[j].Col == t.Col && seen[j].Row == t.Row) {
                    found = 1;
                    break;
                }
            }
            if (!found && seen_count < 128)
                seen[seen_count++] = t;
        }
        /* From tiles (core tiles) */
        for (uint32_t i = 0; i < num_tiles; i++) {
            XAie_LocType t = tiles[i];
            int found = 0;
            for (uint32_t j = 0; j < seen_count; j++) {
                if (seen[j].Col == t.Col && seen[j].Row == t.Row) {
                    found = 1;
                    break;
                }
            }
            if (!found && seen_count < 128)
                seen[seen_count++] = t;
        }
        for (uint32_t i = 0; i < seen_count; i++)
            AieRt_PrintAllBds(dev, seen[i]);
    }

    /* 11. Raw BD register dump for shim tiles (shows stride/wrap/iteration) */
    printf("[AieRt_Debug] ===== Shim Raw BD Registers (stride/wrap) =====\n");
    {
        uint8_t shim_seen[64];
        uint32_t shim_seen_count = 0;
        for (uint32_t i = 0; i < num_ios; i++) {
            if (!s_is_shim(ios[i].tile_loc))
                continue;
            uint8_t col = ios[i].tile_loc.Col;
            int already = 0;
            for (uint32_t j = 0; j < shim_seen_count; j++) {
                if (shim_seen[j] == col) {
                    already = 1;
                    break;
                }
            }
            if (!already && shim_seen_count < 64) {
                shim_seen[shim_seen_count++] = col;
                AieRt_PrintShimBdRawAll(dev, col);
            }
        }
        if (shim_seen_count == 0)
            printf("[AieRt_Debug]  (no shim tiles in ios list)\n");
    }

    /* 12. IO verification */
    AieRt_VerifyIoDescriptors(dev, ios, num_ios);

    /* 13. Kernel klog readback — reads last 2KB of each core tile's DM (offset 0xF800).
     * Shows Aptr/A[0..3] values written by the kernel before acquiring the input window.
     * This reveals whether A data arrived correctly at buf_in_ping_0. */
    printf("[AieRt_Debug] ===== Kernel Log (DM offset 0xF800) =====\n");
    for (uint32_t i = 0; i < num_tiles; i++) {
        if (s_is_core(tiles[i]))
            __Runtime_read_kernel_log(dev, tiles[i]);
    }

    printf("[AieRt_Debug] ============================================================\n\n");
}

/* --------------------------------------------------------------------------
 * Stream switch register debug
 *
 * Reads stream switch master config, slave config, and slave slot registers
 * directly via XAie_Read32, using tile-relative offsets from xaiemlgbl_params.h.
 *
 * Port name tables and counts are sourced directly from xaie2psgbl_params.h.
 * Each tile type has a completely independent port ordering — never mix them.
 *
 * Core tile (CORE_MODULE), base 0x0003F000 / 0x0003F100:
 *   Master[0..22]: CORE0, DMA0-1, CTRL, FIFO0, SOUTH0-3, WEST0-3, NORTH0-5, EAST0-3
 *   Slave[0..24]:  CORE0, DMA0-1, CTRL, FIFO0, SOUTH0-5, WEST0-3, NORTH0-3, EAST0-3,
 *                  AIE_TRACE, MEM_TRACE
 *
 * Shim tile (PL_MODULE), base 0x0003F000 / 0x0003F100:
 *   Master[0..22]: CTRL, FIFO0, SOUTH0-5, WEST0-3, NORTH0-5, EAST0-3, UCONTROLLER
 *   Slave[0..23]:  CTRL, FIFO0, SOUTH0-7, WEST0-3, NORTH0-3, EAST0-3, TRACE, UCONTROLLER
 *
 * MemTile (MEM_TILE_MODULE), base 0x000B0000 / 0x000B0100:
 *   Master[0..16]: DMA0-5, CTRL, SOUTH0-3, NORTH0-5
 *   Slave[0..17]:  DMA0-5, CTRL, SOUTH0-5, NORTH0-3, TRACE
 * -------------------------------------------------------------------------- */

/* --------------------------------------------------------------------------
 * Stream switch port name tables
 *
 * All entries sourced from xaie2psgbl_params.h, listed in register address
 * order (ascending).  Each entry corresponds to one 4-byte register slot.
 * Index 0 = base address, index N = base + N*4.
 * -------------------------------------------------------------------------- */

/* Core tile (CORE_MODULE) master ports — 23 total, base 0x0003F000
 *   XAIE2PSGBL_CORE_MODULE_STREAM_SWITCH_MASTER_CONFIG_AIE_CORE0 = 0x0003F000
 *   ...
 *   XAIE2PSGBL_CORE_MODULE_STREAM_SWITCH_MASTER_CONFIG_EAST3     = 0x0003F058
 */
static const char *s_core_master_names[] = {
    "CORE0",                                                    /* 0:  0x3F000 */
    "DMA0",   "DMA1",                                           /* 1-2: 0x3F004,08 */
    "CTRL",   "FIFO0",                                          /* 3-4: 0x3F00C,10 */
    "SOUTH0", "SOUTH1", "SOUTH2", "SOUTH3",                     /* 5-8: 0x3F014-20 */
    "WEST0",  "WEST1",  "WEST2",  "WEST3",                      /* 9-12: 0x3F024-30 */
    "NORTH0", "NORTH1", "NORTH2", "NORTH3", "NORTH4", "NORTH5", /* 13-18: 0x3F034-48 */
    "EAST0",  "EAST1",  "EAST2",  "EAST3"                       /* 19-22: 0x3F04C-58 */
};
#define AIERT_SS_CORE_MASTER_ACTUAL 23u

/* Core tile (CORE_MODULE) slave ports — 25 total, base 0x0003F100
 *   XAIE2PSGBL_CORE_MODULE_STREAM_SWITCH_SLAVE_CONFIG_AIE_CORE0  = 0x0003F100
 *   ...
 *   XAIE2PSGBL_CORE_MODULE_STREAM_SWITCH_SLAVE_CONFIG_MEM_TRACE  = 0x0003F160
 */
static const char *s_core_slave_names[] = {
    "CORE0",                                                         /* 0:  0x3F100 */
    "DMA0",      "DMA1",                                             /* 1-2: 0x3F104,08 */
    "CTRL",      "FIFO0",                                            /* 3-4: 0x3F10C,10 */
    "SOUTH0",    "SOUTH1",   "SOUTH2", "SOUTH3", "SOUTH4", "SOUTH5", /* 5-10: 0x3F114-28 */
    "WEST0",     "WEST1",    "WEST2",  "WEST3",                      /* 11-14: 0x3F12C-38 */
    "NORTH0",    "NORTH1",   "NORTH2", "NORTH3",                     /* 15-18: 0x3F13C-48 */
    "EAST0",     "EAST1",    "EAST2",  "EAST3",                      /* 19-22: 0x3F14C-58 */
    "AIE_TRACE", "MEM_TRACE"                                         /* 23-24: 0x3F15C,60 */
};
#define AIERT_SS_CORE_SLAVE_ACTUAL 25u

/* --------------------------------------------------------------------------
 * Shim tile (PL_MODULE) stream switch port names
 *
 * DIFFERENT from core tile: no AIE_CORE/DMA on master side; has UCONTROLLER;
 * SOUTH has 6 master outputs but 8 slave inputs (PL interface width).
 *
 * Master ports — 23 total, base 0x0003F000
 *   XAIE2PSGBL_PL_MODULE_STREAM_SWITCH_MASTER_CONFIG_TILE_CTRL   = 0x0003F000
 *   ...
 *   XAIE2PSGBL_PL_MODULE_STREAM_SWITCH_MASTER_CONFIG_UCONTROLLER = 0x0003F058
 *
 * Slave ports — 24 total, base 0x0003F100
 *   XAIE2PSGBL_PL_MODULE_STREAM_SWITCH_SLAVE_CONFIG_TILE_CTRL    = 0x0003F100
 *   ...
 *   XAIE2PSGBL_PL_MODULE_STREAM_SWITCH_SLAVE_CONFIG_UCONTROLLER  = 0x0003F15C
 * -------------------------------------------------------------------------- */

static const char *s_shim_master_names[] = {
    "CTRL",                                                         /* 0:  0x3F000 */
    "FIFO0",                                                        /* 1:  0x3F004 */
    "SOUTH0",     "SOUTH1", "SOUTH2", "SOUTH3", "SOUTH4", "SOUTH5", /* 2-7: 0x3F008-1C */
    "WEST0",      "WEST1",  "WEST2",  "WEST3",                      /* 8-11: 0x3F020-2C */
    "NORTH0",     "NORTH1", "NORTH2", "NORTH3", "NORTH4", "NORTH5", /* 12-17: 0x3F030-44 */
    "EAST0",      "EAST1",  "EAST2",  "EAST3",                      /* 18-21: 0x3F048-54 */
    "UCONTROLLER"                                                   /* 22: 0x3F058 */
};
#define AIERT_SS_SHIM_MASTER_ACTUAL 23u

static const char *s_shim_slave_names[] = {
    "CTRL",                                                                             /* 0:  0x3F100 */
    "FIFO0",                                                                            /* 1:  0x3F104 */
    "SOUTH0",     "SOUTH1", "SOUTH2", "SOUTH3", "SOUTH4", "SOUTH5", "SOUTH6", "SOUTH7", /* 2-9: 0x3F108-24 */
    "WEST0",      "WEST1",  "WEST2",  "WEST3",                                          /* 10-13: 0x3F128-34 */
    "NORTH0",     "NORTH1", "NORTH2", "NORTH3",                                         /* 14-17: 0x3F138-44 */
    "EAST0",      "EAST1",  "EAST2",  "EAST3",                                          /* 18-21: 0x3F148-54 */
    "TRACE",                                                                            /* 22: 0x3F158 */
    "UCONTROLLER"                                                                       /* 23: 0x3F15C */
};
#define AIERT_SS_SHIM_SLAVE_ACTUAL 24u

/* --------------------------------------------------------------------------
 * MemTile (MEM_TILE_MODULE) stream switch port names
 *
 * DIFFERENT from core and shim: no AIE_CORE, no FIFO, no EAST, no WEST;
 * only DMA, CTRL, SOUTH, NORTH connections.
 * SOUTH has 4 master outputs but 6 slave inputs.
 * NORTH has 6 master outputs but 4 slave inputs.
 * Register base is 0x000B0000 (completely separate from core/shim 0x3F000).
 *
 * Master ports — 17 total, base 0x000B0000
 *   XAIE2PSGBL_MEM_TILE_MODULE_STREAM_SWITCH_MASTER_CONFIG_DMA0   = 0x000B0000
 *   ...
 *   XAIE2PSGBL_MEM_TILE_MODULE_STREAM_SWITCH_MASTER_CONFIG_NORTH5 = 0x000B0040
 *
 * Slave ports — 18 total, base 0x000B0100
 *   XAIE2PSGBL_MEM_TILE_MODULE_STREAM_SWITCH_SLAVE_CONFIG_DMA_0   = 0x000B0100
 *   ...
 *   XAIE2PSGBL_MEM_TILE_MODULE_STREAM_SWITCH_SLAVE_CONFIG_TRACE   = 0x000B0144
 * -------------------------------------------------------------------------- */

static const char *s_memtile_master_names[] = {
    "DMA0",   "DMA1",   "DMA2",   "DMA3",   "DMA4",   "DMA5",  /* 0-5:  0xB0000-14 */
    "CTRL",                                                    /* 6:    0xB0018 */
    "SOUTH0", "SOUTH1", "SOUTH2", "SOUTH3",                    /* 7-10: 0xB001C-28 */
    "NORTH0", "NORTH1", "NORTH2", "NORTH3", "NORTH4", "NORTH5" /* 11-16: 0xB002C-40 */
};
#define AIERT_SS_MEMTILE_MASTER_ACTUAL 17u

static const char *s_memtile_slave_names[] = {
    "DMA0",   "DMA1",   "DMA2",   "DMA3",   "DMA4",   "DMA5",   /* 0-5:  0xB0100-14 */
    "CTRL",                                                     /* 6:    0xB0118 */
    "SOUTH0", "SOUTH1", "SOUTH2", "SOUTH3", "SOUTH4", "SOUTH5", /* 7-12: 0xB011C-30 */
    "NORTH0", "NORTH1", "NORTH2", "NORTH3",                     /* 13-16: 0xB0134-40 */
    "TRACE"                                                     /* 17:   0xB0144 */
};
#define AIERT_SS_MEMTILE_SLAVE_ACTUAL 18u

/* Classify a port name into switch type:
 *   "PKT"  — packet-switched port (DMA, CORE, NORTH/SOUTH/EAST/WEST carrying data streams)
 *   "CIRC" — circuit-switched port (same port names but pkt_en=0 on master side)
 *   "CTRL" — control/management port (CTRL, FIFO0, TRACE)
 * For master ports: caller supplies pkt_en from the register.
 * For slave ports: type derived from port name only (no pkt_en bit on slave side).
 */
static const char *s_ss_port_type(const char *name, int pkt_en) {
    /* Control/special ports — never carry DMA data */
    if (strncmp(name, "CTRL", 4) == 0 || strncmp(name, "FIFO", 4) == 0 || strncmp(name, "TRACE", 5) == 0 ||
        strncmp(name, "AIE_TRACE", 9) == 0 || strncmp(name, "MEM_TRACE", 9) == 0)
        return "CTRL";
    /* For master ports, the PACKET_ENABLE bit is definitive */
    if (pkt_en >= 0)
        return pkt_en ? "PKT" : "CIRC";
    /* Slave port — infer from name: DMA/CORE are always packet-switched sources */
    if (strncmp(name, "DMA", 3) == 0 || strncmp(name, "CORE", 4) == 0)
        return "PKT";
    return "CIRC"; /* NORTH/SOUTH/EAST/WEST slave: may carry either, label CIRC */
}

/* Mask semantics note printed for packet-switch slots */
static const char *s_mask_note(uint32_t mask) {
    if (mask == 0x1F)
        return "exact"; /* all 5 PktId bits compared */
    if (mask == 0x00)
        return "wildcard"; /* no bits compared — matches any PktId */
    return "partial";
}

/* Stall interpretation (MEM module local event IDs from xaie_events_aieml.h):
 *   MM2S stall_stream=1 → DMA_MM2S_0_STREAM_BACKPRESSURE (MEM local ID 37): stream switch won't accept packet
 *   S2MM stall_stream=1 → DMA_S2MM_0_STREAM_STARVATION   (MEM local ID 35): waiting for data from stream
 *   stall_lock_acq=1    → DMA_*_STALLED_LOCK              (MEM local IDs 31-34): waiting to acquire lock
 */

void AieRt_PrintStreamSwitchConfig(XAie_DevInst *dev, XAie_LocType tile, int print_all) {
    int is_shim = s_is_shim(tile);
    int is_mem = s_is_memtile(tile);
    uint32_t master_count;
    uint32_t slave_count;
    const char **master_names;
    const char **slave_names;
    const char *tile_type;

    if (is_shim) {
        master_count = AIERT_SS_SHIM_MASTER_ACTUAL;
        slave_count = AIERT_SS_SHIM_SLAVE_ACTUAL;
        master_names = s_shim_master_names;
        slave_names = s_shim_slave_names;
        tile_type = "SHIM";
    } else if (is_mem) {
        master_count = AIERT_SS_MEMTILE_MASTER_ACTUAL;
        slave_count = AIERT_SS_MEMTILE_SLAVE_ACTUAL;
        master_names = s_memtile_master_names;
        slave_names = s_memtile_slave_names;
        tile_type = "MEMTILE";
    } else {
        master_count = AIERT_SS_CORE_MASTER_ACTUAL;
        slave_count = AIERT_SS_CORE_SLAVE_ACTUAL;
        master_names = s_core_master_names;
        slave_names = s_core_slave_names;
        tile_type = "CORE";
    }

    u64 tile_base = ((u64)tile.Col << 25) | ((u64)tile.Row << 20);

    /* Select register bases according to tile type */
    uint32_t ss_master_base = is_mem ? AIERT_SS_MEM_MASTER_BASE : AIERT_SS_MASTER_BASE;
    uint32_t ss_slave_base = is_mem ? AIERT_SS_MEM_SLAVE_BASE : AIERT_SS_SLAVE_BASE;
    uint32_t ss_slot_base = is_mem ? AIERT_SS_MEM_SLOT_BASE : AIERT_SS_SLOT_BASE;

    /* ------------------------------------------------------------------ */
    /* JSON output                                                          */
    /* ------------------------------------------------------------------ */
    printf("{\n");
    printf("  \"tile\": {\"col\": %u, \"row\": %u, \"type\": \"%s\"},\n", (unsigned)tile.Col, (unsigned)tile.Row,
           tile_type);
    printf("  \"masters\": [\n");

    int first_master = 1;
    for (uint32_t i = 0; i < master_count; i++) {
        u64 reg_off = tile_base | (ss_master_base + i * 4u);
        u32 val = 0;
        AieRC rc = XAie_Read32(dev, reg_off, &val);
        if (rc != XAIE_OK)
            continue;

        int enabled = (val & AIERT_SS_MSTR_ENABLE_MASK) ? 1 : 0;
        int pkt_en = (val & AIERT_SS_MSTR_PKT_EN_MASK) ? 1 : 0;
        int drop_hdr = (val & AIERT_SS_MSTR_DROP_MASK) ? 1 : 0;
        uint32_t config = val & AIERT_SS_MSTR_CONFIG_MASK;

        if (!print_all && !enabled)
            continue;

        const char *port_type = s_ss_port_type(master_names[i], pkt_en);

        /* drop_header note: only meaningful for PKT masters */
        const char *drop_note = "";
        if (pkt_en && drop_hdr)
            drop_note = "DROP_HEADER (strip pkt hdr before forwarding — last PKT-SW hop)";
        if (pkt_en && !drop_hdr)
            drop_note = "DONOT_DROP_HEADER (preserve pkt hdr for downstream PKT-SW slave)";

        if (!first_master)
            printf(",\n");
        first_master = 0;

        printf("    {\n");
        printf("      \"index\": %u,\n", (unsigned)i);
        printf("      \"port\": \"%s\",\n", master_names[i]);
        printf("      \"switch_type\": \"%s\",\n", port_type);
        printf("      \"enabled\": %s,\n", enabled ? "true" : "false");
        printf("      \"packet_switch\": %s,\n", pkt_en ? "true" : "false");
        printf("      \"drop_header\": %s,\n", drop_hdr ? "true" : "false");
        if (pkt_en)
            printf("      \"drop_header_note\": \"%s\",\n", drop_note);
        printf("      \"config\": \"0x%02X\",\n", (unsigned)config);
        printf("      \"raw\": \"0x%08X\"\n", (unsigned)val);
        printf("    }");
    }
    printf("\n  ],\n");

    /* --- Slave config + slot registers --- */
    printf("  \"slaves\": [\n");
    int first_slave = 1;
    for (uint32_t i = 0; i < slave_count; i++) {
        u64 slv_cfg_off = tile_base | (ss_slave_base + i * 4u);
        u32 slv_val = 0;
        AieRC rc = XAie_Read32(dev, slv_cfg_off, &slv_val);
        if (rc != XAIE_OK)
            continue;

        int slv_en = (slv_val & AIERT_SS_SLV_ENABLE_MASK) ? 1 : 0;

        u32 slots[4] = {0, 0, 0, 0};
        int slot_en[4] = {0, 0, 0, 0};
        int any_slot = 0;
        for (uint32_t s = 0; s < 4u; s++) {
            u64 slot_off = tile_base | (ss_slot_base + i * 16u + s * 4u);
            AieRC src = XAie_Read32(dev, slot_off, &slots[s]);
            if (src == XAIE_OK && (slots[s] & AIERT_SS_SLOT_ENABLE_MASK)) {
                slot_en[s] = 1;
                any_slot = 1;
            }
        }

        if (!print_all && !slv_en && !any_slot)
            continue;

        /* Slave port type: use pkt_en=-1 (no register bit on slave side) */
        const char *port_type = s_ss_port_type(slave_names[i], -1);
        /* Refine: if any slot is enabled, it must be packet-switched */
        if (any_slot && strcmp(port_type, "CTRL") != 0)
            port_type = "PKT";

        if (!first_slave)
            printf(",\n");
        first_slave = 0;

        printf("    {\n");
        printf("      \"index\": %u,\n", (unsigned)i);
        printf("      \"port\": \"%s\",\n", slave_names[i]);
        printf("      \"switch_type\": \"%s\",\n", port_type);
        printf("      \"enabled\": %s,\n", slv_en ? "true" : "false");
        printf("      \"raw\": \"0x%08X\",\n", (unsigned)slv_val);
        printf("      \"slots\": [\n");

        int first_slot = 1;
        for (uint32_t s = 0; s < 4u; s++) {
            if (!print_all && !slot_en[s])
                continue;
            uint32_t pkt_id = (slots[s] & AIERT_SS_SLOT_PKTID_MASK) >> AIERT_SS_SLOT_PKTID_LSB;
            uint32_t pkt_msk = (slots[s] & AIERT_SS_SLOT_MASK_MASK) >> AIERT_SS_SLOT_MASK_LSB;
            uint32_t msel = (slots[s] & AIERT_SS_SLOT_MSEL_MASK) >> AIERT_SS_SLOT_MSEL_LSB;
            uint32_t arb = (slots[s] & AIERT_SS_SLOT_ARB_MASK) >> AIERT_SS_SLOT_ARB_LSB;
            /* mask=0x1F means exact PktId match required (MEM local ID 37 fires if mismatch);
             * mask=0x00 means wildcard — accepts any PktId (used at merge points) */
            const char *match_note = s_mask_note(pkt_msk);

            if (!first_slot)
                printf(",\n");
            first_slot = 0;
            printf("        {\n");
            printf("          \"slot\": %u,\n", (unsigned)s);
            printf("          \"enabled\": %s,\n", slot_en[s] ? "true" : "false");
            printf("          \"pkt_id\": %u,\n", (unsigned)pkt_id);
            printf("          \"pkt_mask\": \"0x%02X\",\n", (unsigned)pkt_msk);
            printf("          \"match_mode\": \"%s\",\n", match_note);
            /* match_mode note: exact match means wrong pkt_id → DMA_MM2S_0_STREAM_BACKPRESSURE (MEM local ID 37) */
            if (pkt_msk == 0x1F)
                printf("          \"match_note\": \"exact: DMA BD pkt_id must equal %u or "
                       "DMA_MM2S_0_STREAM_BACKPRESSURE (MEM local ID 37) fires\",\n",
                       (unsigned)pkt_id);
            else if (pkt_msk == 0x00)
                printf("          \"match_note\": \"wildcard: any pkt_id accepted (merge point)\",\n");
            printf("          \"msel\": %u,\n", (unsigned)msel);
            printf("          \"arb\": %u,\n", (unsigned)arb);
            printf("          \"raw\": \"0x%08X\"\n", (unsigned)slots[s]);
            printf("        }");
        }
        printf("\n      ]\n");
        printf("    }");
    }
    printf("\n  ]\n");
    printf("}\n");
}

void AieRt_PrintStreamSwitchConfigAll(XAie_DevInst *dev, const XAie_LocType *tiles, uint32_t num_tiles, int print_all) {
    printf("[AieRt_SS] ===== Stream Switch Config (%u tiles) =====\n", (unsigned)num_tiles);
    printf("[\n");
    for (uint32_t i = 0; i < num_tiles; i++) {
        AieRt_PrintStreamSwitchConfig(dev, tiles[i], print_all);
        if (i + 1 < num_tiles)
            printf(",\n");
    }
    printf("]\n");
}

/* --------------------------------------------------------------------------
 * Coordinate-based debug snapshot helper
 * -------------------------------------------------------------------------- */

/* Max IOs and tiles supported by the stack-allocated snapshot helper.
 * Baremetal environments often have broken/limited calloc (nosys.specs),
 * so we avoid heap allocation entirely. */
#define AIERT_SNAPSHOT_MAX_IOS 128
#define AIERT_SNAPSHOT_MAX_TILES 64

void AieRt_DebugSnapshotFromCoords(XAie_DevInst *dev, const uint8_t *io_cols, const uint8_t *io_rows,
                                   const uint8_t *io_chs, const uint8_t *io_bds, const int *io_dirs, uint32_t num_ios,
                                   const uint8_t *tile_cols, const uint8_t *tile_rows, uint32_t num_tiles) {
    /* Stack-allocated arrays — avoids calloc which fails on baremetal */
    struct_io ios[AIERT_SNAPSHOT_MAX_IOS];
    XAie_LocType tiles[AIERT_SNAPSHOT_MAX_TILES];

    if (num_ios > AIERT_SNAPSHOT_MAX_IOS) {
        printf("[AieRt_Debug] WARNING: num_ios=%u exceeds max %d, clamping\n", (unsigned)num_ios,
               AIERT_SNAPSHOT_MAX_IOS);
        num_ios = AIERT_SNAPSHOT_MAX_IOS;
    }
    if (num_tiles > AIERT_SNAPSHOT_MAX_TILES) {
        printf("[AieRt_Debug] WARNING: num_tiles=%u exceeds max %d, clamping\n", (unsigned)num_tiles,
               AIERT_SNAPSHOT_MAX_TILES);
        num_tiles = AIERT_SNAPSHOT_MAX_TILES;
    }

    memset(ios, 0, num_ios * sizeof(struct_io));
    for (uint32_t i = 0; i < num_ios; i++) {
        ios[i].tile_loc = XAie_TileLoc(io_cols[i], io_rows[i]);
        ios[i].channel_id = io_chs[i];
        ios[i].bd_id = io_bds[i];
        ios[i].direction = (XAie_DmaDirection)io_dirs[i];
    }

    for (uint32_t i = 0; i < num_tiles; i++) {
        tiles[i] = XAie_TileLoc(tile_cols[i], tile_rows[i]);
    }

    AieRt_DebugSnapshot(dev, ios, num_ios, tiles, num_tiles);
}

/* --------------------------------------------------------------------------
 * Performance counter debug
 * -------------------------------------------------------------------------- */

void AieRt_PrintCoreTilePerfCounters(XAie_DevInst *dev, XAie_LocType tile) {
    if (!s_is_core(tile))
        return;

    uint32_t val0 = 0, val1 = 0;
    AieRC rc0 = XAie_PerfCounterGet(dev, tile, XAIE_MEM_MOD, 0, &val0);
    AieRC rc1 = XAie_PerfCounterGet(dev, tile, XAIE_MEM_MOD, 1, &val1);

    printf("[AieRt_Debug] PerfCnt tile(%u,%u) MEM_MOD: "
           "cnt0=%u (rc=%d) cnt1=%u (rc=%d)\n",
           (unsigned)tile.Col, (unsigned)tile.Row, val0, (int)rc0, val1, (int)rc1);
}

void AieRt_PrintCoreTilePerfCountersAll(XAie_DevInst *dev, const XAie_LocType *tiles, uint32_t num_tiles) {
    printf("[AieRt_Debug] ===== Core Tile Perf Counters (MEM_MOD cnt0/cnt1) =====\n");
    for (uint32_t i = 0; i < num_tiles; i++) {
        if (s_is_core(tiles[i]))
            AieRt_PrintCoreTilePerfCounters(dev, tiles[i]);
    }
}

/* ==========================================================================
 * Shim DMA Loopback Debug API
 *
 * Performs a DDR→ShimDMA→DDR loopback test on a single shim tile column.
 * Flow:
 *   1. Allocate two DDR buffers (src, dst) via XAie_MemAllocate.
 *   2. Fill src with a known pattern, clear dst.
 *   3. Configure shim tile stream switch: circuit-switch SOUTH port 3
 *      back to itself (MM2S output → S2MM input on same port).
 *   4. Set up two BDs on the shim tile:
 *        BD 0 — MM2S: reads src buffer from DDR
 *        BD 1 — S2MM: writes received data to dst buffer in DDR
 *   5. Push BDs to channel queues, enable channels, poll for completion.
 *   6. Compare src vs dst; print PASS/FAIL.
 *   7. Free DDR buffers.
 *
 * Reference: aieml-tests/src/xaie_shimdma_loopback.c
 * ========================================================================== */

int AieRt_ShimDmaLoopback(XAie_DevInst *dev, uint8_t col, uint32_t *srcaddr, uint32_t *dstaddr, uint32_t len) {
    AieRC RC;
    uint64_t src_addr = (uint64_t)(uintptr_t)srcaddr;
    uint64_t dst_addr = (uint64_t)(uintptr_t)dstaddr;
    uint32_t num_words = len / sizeof(uint32_t);

    printf("[AieRt_ShimDmaLoopback] col=%u src=0x%lx dst=0x%lx len=%u\n", (unsigned)col, (unsigned long)src_addr,
           (unsigned long)dst_addr, (unsigned)len);

    /* ---- 1. Configure stream switch: SOUTH3 → SOUTH3 circuit loopback ---- */
    XAie_LocType shim_tile = XAie_TileLoc(col, 0);

    RC = XAie_StrmConnCctEnable(dev, shim_tile, SOUTH, 3, SOUTH, 3);
    printf("[AieRt_ShimDmaLoopback] StrmConnCctEnable SOUTH3→SOUTH3 rc=%d\n", (int)RC);

    RC = XAie_EnableShimDmaToAieStrmPort(dev, shim_tile, 3);
    printf("[AieRt_ShimDmaLoopback] EnableShimDmaToAieStrmPort(3) rc=%d\n", (int)RC);

    RC = XAie_EnableAieToShimDmaStrmPort(dev, shim_tile, 3);
    printf("[AieRt_ShimDmaLoopback] EnableAieToShimDmaStrmPort(3) rc=%d\n", (int)RC);

    /* ---- 2. Configure BDs ----
     *   BD 0: MM2S — read from src DDR buffer
     *   BD 1: S2MM — write to dst DDR buffer
     */
    XAie_DmaDesc DmaDescMM2S, DmaDescS2MM;

    /* MM2S BD (BD 0): read src from DDR */
    RC = XAie_DmaDescInit(dev, &DmaDescMM2S, shim_tile);
    RC = XAie_DmaSetAddrLen(&DmaDescMM2S, src_addr, len);
    RC = XAie_DmaEnableBd(&DmaDescMM2S);
    RC = XAie_DmaSetAxi(&DmaDescMM2S, 0U, 16U, 0U, 0U, 0U);
    RC = XAie_DmaWriteBd(dev, &DmaDescMM2S, shim_tile, 0);
    printf("[AieRt_ShimDmaLoopback] MM2S BD0 addr=0x%lx len=%u rc=%d\n", (unsigned long)src_addr, (unsigned)len,
           (int)RC);

    /* S2MM BD (BD 1): write dst to DDR */
    RC = XAie_DmaDescInit(dev, &DmaDescS2MM, shim_tile);
    RC = XAie_DmaSetAddrLen(&DmaDescS2MM, dst_addr, len);
    RC = XAie_DmaEnableBd(&DmaDescS2MM);
    RC = XAie_DmaSetAxi(&DmaDescS2MM, 0U, 16U, 0U, 0U, 0U);
    RC = XAie_DmaWriteBd(dev, &DmaDescS2MM, shim_tile, 1);
    printf("[AieRt_ShimDmaLoopback] S2MM BD1 addr=0x%lx len=%u rc=%d\n", (unsigned long)dst_addr, (unsigned)len,
           (int)RC);

    /* ---- 3. Push BDs to queues and enable channels ----
     *   AIEML shim DMA channel-to-port mapping (fixed in HW):
     *     MM2S ch0 → SOUTH slave port 3   (3, 7)
     *     S2MM ch1 → SOUTH master port 3  (1, 3)
     *   We configured circuit-switch on port 3, so must use these channels.
     *   S2MM must be armed FIRST (receiver ready before sender starts).
     */
    RC = XAie_DmaChannelPushBdToQueue(dev, shim_tile, 1, DMA_S2MM, 1);
    printf("[AieRt_ShimDmaLoopback] S2MM ch1 push BD1 rc=%d\n", (int)RC);

    RC = XAie_DmaChannelPushBdToQueue(dev, shim_tile, 0, DMA_MM2S, 0);
    printf("[AieRt_ShimDmaLoopback] MM2S ch0 push BD0 rc=%d\n", (int)RC);

    RC = XAie_DmaChannelEnable(dev, shim_tile, 1, DMA_S2MM);
    printf("[AieRt_ShimDmaLoopback] S2MM ch1 enable rc=%d\n", (int)RC);

    RC = XAie_DmaChannelEnable(dev, shim_tile, 0, DMA_MM2S);
    printf("[AieRt_ShimDmaLoopback] MM2S ch0 enable rc=%d\n", (int)RC);

    /* ---- 4. Poll for S2MM ch1 completion ---- */
    u8 pending = 1;
    uint32_t poll_count = 0;
    const uint32_t max_polls = 10000;
    while (pending != 0 && poll_count < max_polls) {
        RC = XAie_DmaGetPendingBdCount(dev, shim_tile, 1, DMA_S2MM, &pending);
        if (RC != XAIE_OK) {
            printf("[AieRt_ShimDmaLoopback] ERROR: GetPendingBdCount failed rc=%d\n", (int)RC);
            break;
        }
        poll_count++;
    }

    if (pending != 0) {
        printf("[AieRt_ShimDmaLoopback] TIMEOUT: S2MM still pending after %u polls\n", (unsigned)poll_count);
        AieRt_PrintShimDmaStatus(dev, col);
        return -2;
    }

    printf("[AieRt_ShimDmaLoopback] DMA transfer complete after %u polls\n", (unsigned)poll_count);

    /* ---- 5. Verify: compare src vs dst ---- */
    int errors = 0;
    for (uint32_t i = 0; i < num_words; i++) {
        if (srcaddr[i] != dstaddr[i]) {
            if (errors < 16) {
                printf("[AieRt_ShimDmaLoopback] MISMATCH [%u]: src=0x%08x dst=0x%08x\n", (unsigned)i,
                       (unsigned)srcaddr[i], (unsigned)dstaddr[i]);
            }
            errors++;
        }
    }

    if (errors == 0) {
        printf("[AieRt_ShimDmaLoopback] PASS col=%u (%u words verified)\n", (unsigned)col, (unsigned)num_words);
    } else {
        printf("[AieRt_ShimDmaLoopback] FAIL col=%u (%d/%u mismatches)\n", (unsigned)col, errors, (unsigned)num_words);
    }

    /* Print first 16 words of src and dst for visual inspection */
    uint32_t print_n = (num_words < 16) ? num_words : 16;
    printf("[AieRt_ShimDmaLoopback] src[0..%u]:", (unsigned)(print_n - 1));
    for (uint32_t i = 0; i < print_n; i++)
        printf(" 0x%08x", (unsigned)srcaddr[i]);
    printf("\n");
    printf("[AieRt_ShimDmaLoopback] dst[0..%u]:", (unsigned)(print_n - 1));
    for (uint32_t i = 0; i < print_n; i++)
        printf(" 0x%08x", (unsigned)dstaddr[i]);
    printf("\n");

    return errors == 0 ? 0 : -3;
}

int AieRt_ShimDmaLoopbackAllCols(XAie_DevInst *dev, const uint8_t *cols, int num_cols, uint32_t *srcaddr,
                                 uint32_t *dstaddr, uint32_t len) {
    printf("[AieRt_ShimDmaLoopback] ===== Running loopback on %d columns, len=%u =====\n", num_cols, (unsigned)len);
    int total_fail = 0;
    for (int i = 0; i < num_cols; i++) {
        int rc = AieRt_ShimDmaLoopback(dev, cols[i], srcaddr, dstaddr, len);
        if (rc != 0) {
            printf("[AieRt_ShimDmaLoopback] Column %u FAILED (rc=%d)\n", (unsigned)cols[i], rc);
            total_fail++;
        }
    }
    printf("[AieRt_ShimDmaLoopback] ===== Result: %d/%d columns passed =====\n", num_cols - total_fail, num_cols);
    return total_fail;
}
