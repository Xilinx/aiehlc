
/* the step every offloaded conv call site runs (aiegraph group step, "args" ABI) */
extern "C" int @K@_aie_conv(int cid, void *args, int nargs, int sh, int sw, int ph, int pw, int act, int dw, int np,
                            int kk, int kp, int mc, int kch) {
    g_uc_calls++;
    const int rc = uc_chunk_conv((void *const *)args, nargs, sh, sw, ph, pw, act, dw, np, kk, kp, mc, kch);
    if (rc != 0) {
        printf("[@K@_aie_conv] layer %d rc=%d\n", cid, rc);
        if (!g_uc_rc)
            g_uc_rc = rc;
    }
    return rc;
}

// Carrier entry for aiegraph's offload flow (builds + links this archive). Never called: the generated
// graph_driver.c calls @K@_aie_conv() at every offloaded conv call site, including this layer's.
AIEK_ENTRY(@E@) {
    (void)args;
    (void)num_args;
    printf("[@E@] carrier entry called -- driver not patched\n");
    if (!g_uc_rc)
        g_uc_rc = -7;
    return AIEK_ERR_KERNEL;
}

@SPEC@
