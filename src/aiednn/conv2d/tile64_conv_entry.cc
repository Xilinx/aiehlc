    return 0;
}

extern "C" int aiednn_aie_conv(int cid, void *args, int nargs, const uint16_t *wbf, int sh, int sw, int ph, int pw) {
    g_uc_calls++;
    const int rc = uc_conv((void *const *)args, nargs, wbf, sh, sw, ph, pw);
    if (rc != 0) {
        printf("[aiednn_aie_conv] layer %d rc=%d\n", cid, rc);
        if (!g_uc_rc)
            g_uc_rc = rc;
    }
    return rc;
}

// Carrier entry for aiegraph's offload flow (builds + links this archive). Never called:
// graph_driver.c (post-processed by build_unet.py) calls aiednn_aie_conv() at every
// partition call site, including this layer's.
AIEK_ENTRY(@E@) {
    (void)args;
    (void)num_args;
    printf("[@E@] carrier entry called -- driver not patched\n");
    if (!g_uc_rc)
        g_uc_rc = -7;
    return AIEK_ERR_KERNEL;
}

@SPEC@
