
#if defined(__chess__)
    { // --launch-opt warm (--skip-bss): softfloat's .bss float_rounding_mode is read by
      // float32_add/mul and never written by anyone -> zero it on every launch
        extern int float_rounding_mode;
        float_rounding_mode = 0;
    }
#endif
