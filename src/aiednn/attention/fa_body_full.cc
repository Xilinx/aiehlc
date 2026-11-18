        const int16_t *kb = (const int16_t *)aie::load_buffer(v_@K@, {pn, (2 * b) * @E@}, {1, @E@});
        @NS@::qkb(kb, b, 0);
        aie::release(v_@K@);
        @NS@::softb(b);
        const int16_t *vb = (const int16_t *)aie::load_buffer(v_@K@, {pn, (2 * b + 1) * @E@}, {1, @E@});
        @NS@::pvb(vb);
        aie::release(v_@K@);