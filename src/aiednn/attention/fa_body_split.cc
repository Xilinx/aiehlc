        const int16_t *k0 = (const int16_t *)aie::load_buffer(v_@K@, {pn, (4 * b) * @E@}, {1, @E@});
        @NS@::qkb(k0, b, 0);
        aie::release(v_@K@);
        const int16_t *k1 = (const int16_t *)aie::load_buffer(v_@K@, {pn, (4 * b + 1) * @E@}, {1, @E@});
        @NS@::qkb(k1, b, 1);
        aie::release(v_@K@);
        @NS@::softb(b);
        const int16_t *v0 = (const int16_t *)aie::load_buffer(v_@K@, {pn, (4 * b + 2) * @E@}, {1, @E@});
        if ((pm & 1) == 0)
            @NS@::pvb(v0);
        aie::release(v_@K@);
        const int16_t *v1 = (const int16_t *)aie::load_buffer(v_@K@, {pn, (4 * b + 3) * @E@}, {1, @E@});
        if ((pm & 1) == 1)
            @NS@::pvb(v1);
        aie::release(v_@K@);