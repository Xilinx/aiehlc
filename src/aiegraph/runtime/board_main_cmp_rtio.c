    int rc;
    if (@M@_rtio_own)
        @CALL@
    else {
        printf("CMP @M@ skipped runtime-input aie_launches=%d run_rc=%d\n", @LAUNCHES@, @RUNRC@);
        fflush(stdout);
        rc = @RUNRC@ != 0;
    }
    if (aiehlc_io_dump_out())
        aiehlc_io_dump("@M@", out, (unsigned long)n * sizeof(float), (unsigned long)n);
