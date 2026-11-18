#ifdef @MU@_LAUNCH_WARM
    {
        double aie_ms = 0.0;
        int kloads = 0;
        unsigned cold_tiles = 0;
        int wl = 0;
        unet_aie_stats(&wl, 0, 0, 0);
        unet_aie_launch_stats(&aie_ms, &kloads, &cold_tiles);
        printf("LAUNCH warm ms_per_launch=%.3f launches=%d aie_wall_ms=%.3f kloads=%d opts=%s,cold_tile_loads=%u"
               " ctrlpkt=%s aot_fallbacks=%llu\n",
               wl ? aie_ms / wl : 0.0, wl, aie_ms, kloads, @MU@_LAUNCH_OPTS, cold_tiles, @MU@_CTRLPKT,
               __Runtime_ctrl_aot_fallbacks());
        /* P1: per-launch split of aie_wall_ms by runtime phase (A78 pmccntr, AIEHLC_PROFILING=1 runtime).
         * PH_COREEN is the A78 cost of the core-enable writes, not AIE compute: the AIE compute is
         * inside wait_io (completion = output-DMA drain). core_reset is part of kload. */
        unsigned long long ph[4] = {0, 0, 0, 0}, wio = 0, kel = 0, krst = 0, plan = 0, sync = 0;
        unsigned int phn[4] = {0, 0, 0, 0}, wion = 0, keln = 0, krstn = 0, plann = 0, syncn = 0;
        __Runtime_phase_cycles(ph, phn); /* PH_KLOAD=0 PH_BDCFG=1 PH_COREEN=2 PH_STARTIO=3 */
        __Runtime_wait_io_cycles(&wio, &wion);
        __Runtime_kload_split_cycles(&kel, &keln, &krst, &krstn);
        __Runtime_setup_split_cycles(&plan, &plann, &sync, &syncn);
        const double g_s = (double)(t1 - t0) / (double)COUNTS_PER_SECOND;
        const unsigned long long pmu_hz = g_s > 0.0 ? (unsigned long long)((double)(pmc1 - pmc0) / g_s + 0.5) : 0ULL;
        const double cms = pmu_hz ? 1000.0 / (double)pmu_hz : 0.0; /* ms per PMU cycle */
        const double nl = wl ? (double)wl : 1.0;
        /* app-side launch regions (pmccntr, per launch, see unet_aie.py HOST_CC): host->DMA flush joins
         * startio, DMA->host invalidate joins wait_io, the generated-host glue (lock init, createio, dev
         * sync, commit: gemm_bf16 call minus runtime phases) joins bdcfg */
        unsigned long long sp_pre = 0, sp_call = 0, sp_glue = 0, sp_post = 0;
        unet_aie_split_stats(&sp_pre, &sp_call, &sp_glue, &sp_post);
        const unsigned long long c_kl = ph[0], c_bd = ph[1] + sp_glue, c_si = ph[3] + sp_pre, c_ce = ph[2],
                                 c_wi = wio + sp_post;
        const double rt_ms = (double)(ph[0] + ph[1] + ph[2] + ph[3] + wio) * cms;
        const double sum_ms = (double)(c_kl + c_bd + c_si + c_ce + c_wi) * cms;
        printf("LAUNCH split kload=%.3f bdcfg=%.3f startio=%.3f coreen=%.3f wait_io=%.3f core_reset=%.3f "
               "ms_per_launch=%.3f pmu_hz=%llu sum_vs_aie_wall=%.3f\n",
               c_kl * cms / nl, c_bd * cms / nl, c_si * cms / nl, c_ce * cms / nl, c_wi * cms / nl,
               krst * cms / nl, aie_ms / nl, pmu_hz, aie_ms > 0.0 ? sum_ms / aie_ms : 0.0);
        printf("LAUNCH split_detail calls kload=%u bdcfg=%u startio=%u coreen=%u wait_io=%u core_reset=%u "
               "kload_elf_ms=%.3f plan_ms=%.3f sync_ms=%.3f rt_bdcfg=%.4f rt_startio=%.4f rt_wait_io=%.4f "
               "host_glue_bdcfg=%.4f host_flush_startio=%.4f host_inval_wait_io=%.4f gemm_call=%.4f "
               "rt_phase_sum_ms=%.3f phase_sum_ms=%.3f aie_wall_ms=%.3f "
               "unattributed_ms_per_launch=%.4f pmc_delta=%llu xtime_s=%.3f\n",
               phn[0], phn[1], phn[3], phn[2], wion, krstn, kel * cms / nl, plan * cms / nl, sync * cms / nl,
               ph[1] * cms / nl, ph[3] * cms / nl, wio * cms / nl, sp_glue * cms / nl, sp_pre * cms / nl,
               sp_post * cms / nl, sp_call * cms / nl, rt_ms, sum_ms, aie_ms, (aie_ms - sum_ms) / nl, pmc1 - pmc0,
               g_s);
        fflush(stdout);
    }
#endif
