    {
        double aie_ms_p = 0.0;
        unet_aie_launch_stats(&aie_ms_p, 0, 0);
        unet_prof_print(1000.0 * (double)(t1 - t0) / (double)COUNTS_PER_SECOND, aie_ms_p);
        fflush(stdout);
    }
