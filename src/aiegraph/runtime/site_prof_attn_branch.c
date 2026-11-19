        if (is_aie && strncmp(up_tag[i], "aie aiednn_aie_attn ", 20) == 0) /* GATT attention bmm site */
            printf("PROF part=%u op=aie_host_attn%s ms=%.3f bf16_expand_ms=%.3f site_ms=%.3f aie_ms=%.3f\n", i,
                   up_tag[i] + 20, a78 + e, e, site, up_aie_ms[i]);
        else 