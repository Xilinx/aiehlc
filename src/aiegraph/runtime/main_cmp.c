/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/
int launches = 0, run_rc = -99, calls = 0;
unsigned long long macs = 0;
aiednn_aie_stats(&launches, &run_rc, &calls, &macs);
unsigned long long ph_cyc[4] = {0, 0, 0, 0};
unsigned int ph_calls[4] = {0, 0, 0, 0};
__Runtime_phase_cycles(ph_cyc, ph_calls); /* [2] = PH_COREEN (AIEHLC_PROFILING=1 runtime) */
const double frac = (double)macs / (double)@MU @_TOTAL_GEMM_MACS;
printf("AIE @M@: aiednn_aie_conv calls=%d (planned %d) run_rc=%d aie_launches=%d (planned %d) "
       "PH_COREEN calls=%u cycles=%llu aie_macs=%llu aie_flop_frac=%.4f\n",
       calls, @MU @_AIE_PLANNED_CALLS, run_rc, launches, @MU @_AIE_PLANNED_LAUNCHES, ph_calls[2], ph_cyc[2], macs,
       frac);
/*KT+*/    {
        unsigned long long kt_max = 0, kt_all = 0;
        int kt_n = 0, kt_bad = 0;
        aiednn_aie_ktimer_stats(&kt_max, &kt_all, &kt_n, &kt_bad);
        /* AIE core timer, read by the kernel at entry/exit, summed over EVERY launch (max of the 16 cores) */
        printf("KTIMER core_cycles_per_launch=%llu launches=%d read=%d bad_tails=%d mean_core_cycles=%.1f "
               "total_max_core_cycles=%llu\n", launches ? kt_max / (unsigned long long)launches : 0ULL, launches,
               kt_n, kt_bad, kt_n ? (double)kt_all / (16.0 * kt_n) : 0.0, kt_max);
    }
/*KT-*/    __Runtime_ctrl_shim_bd_ctrl_stats_print(); /* [SHIM_BD_CTRL] + [CTRLPKT_AOT] in every build */
    int rc = p1_cmp("@M@", out, "bf16", launches, (int)ph_calls[2], run_rc, frac);
