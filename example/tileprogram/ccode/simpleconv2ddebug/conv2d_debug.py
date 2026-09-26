#!/usr/bin/env python3
"""
conv2d_debug.py — simulate simpleconv2d.cc input/filter DDR layout and replay the
conv2d_spatial kernel for the FIRST core tile (mesh row 0, col 0, on-core round 0).

Goal (from the debug request):
  * Reproduce, byte-for-byte, the host input-value layout and filter-value layout
    that simpleconv2d.cc / simpleconv2d.h materialize in DDR.
  * Extract the FIRST input slab  = [H=19, W=61, C=4]  (TILE_H x TILE_W x SP_C).
  * Use the FIRST 16 of 64 filters (OC_PER_G), each of size 7*7*4 = 196 (K).
  * Run the exact on-chip im2col + matmul the kernel does and print every input
    value used, every filter value used, and the resulting output tile.

All arithmetic mirrors the int8/int16 semantics of the AIE kernel:
  input/filter are int8_t; products accumulate in int16_t; the result is clamped
  to [-128, 127] and stored as int8_t.

Value layout note:
  To make the output tile land in a normal (bell-shaped) distribution centered
  near 0 instead of saturating at 127/-128, the input and filter values are
  small zero-mean samples:
    * input  = mix32(real_idx + INPUT_SEED)  % 9 - 4  -> int in [-4, 4]
    * filter = mix32(filt_key + FILTER_SEED) % 3 - 1  -> int in {-1, 0, 1}
  (real channels only; the pad channel and spatial border stay 0)
  Summing ~147 zero-mean products (49 taps x 3 real channels) yields, by the
  central limit theorem, an ~N(0, sigma^2) accumulator with sigma ~= 24, which
  fits inside [-128, 127] so clamping is rare.

  The values are produced by mix32() — the SAME index-keyed integer hash and
  seeds used by simpleconv2d.cc's host init — so this replay reproduces the
  hardware input/filter (and therefore the output) BYTE-FOR-BYTE.
"""

# Seeds and hash MUST match simpleconv2d.cc (INPUT_SEED/FILTER_SEED + mix32).
INPUT_SEED  = 1234
FILTER_SEED = 5678


def mix32(x):
    """lowbias32 integer hash — bit-identical to mix32() in simpleconv2d.cc."""
    x &= 0xFFFFFFFF
    x ^= x >> 16
    x = (x * 0x7feb352d) & 0xFFFFFFFF
    x ^= x >> 15
    x = (x * 0x846ca68b) & 0xFFFFFFFF
    x ^= x >> 16
    return x & 0xFFFFFFFF

# ─────────────────────────────────────────────────────────────────────────────
# Constants (mirrored EXACTLY from simpleconv2d.h and simpleconv2d.cc)
# ─────────────────────────────────────────────────────────────────────────────
INPUT_H       = 224
INPUT_W       = 224
INPUT_C       = 3          # real (semantic) channels
INPUT_C_ALIGN = 4          # channel-layout stride (cin padded with a zero channel)
KERNEL_H      = 7
KERNEL_W      = 7
NUM_FILTERS   = 64
STRIDE        = 2
PAD           = 3

INPUT_H_PAD   = INPUT_H + 2 * PAD    # 230
INPUT_W_PAD   = INPUT_W + 2 * PAD    # 230  (row pitch, in pixels)
K             = KERNEL_H * KERNEL_W * INPUT_C_ALIGN   # 196

# Kernel-visible SP_* geometry (spatial-halo path) from simpleconv2d.cc
SP_KH   = 7
SP_KW   = 7
SP_C    = 4
SP_S    = 2
OH_T    = 7    # output-tile height (== oh_per_row == SP_OHR)
OW_T    = 28   # output-tile width  (== ow_dim)
TILE_H  = (OH_T - 1) * SP_S + SP_KH          # 19 input rows per on-core round
TILE_W  = (OW_T - 1) * SP_S + SP_KW          # 61 input cols per on-core round
OC_PER_G = 16  # filters per tile column (first 16 of 64)

# Kernel locals
oh_per_row = OH_T          # 7
ow_dim     = OW_T          # 28
k_dim      = SP_KH * SP_KW * SP_C   # 196
raw_wc     = TILE_W * SP_C          # 244 (per-chunk input-row width in the slab)
tile_cols  = OC_PER_G               # 16

# Full (whole-image, all-filter) output geometry — what the complete conv2d
# produces across every mesh tile / round / filter group combined.
OUT_H = (INPUT_H + 2 * PAD - KERNEL_H) // STRIDE + 1   # 112
OUT_W = (INPUT_W + 2 * PAD - KERNEL_W) // STRIDE + 1   # 112
OUT_C = NUM_FILTERS                                    # 64


# ─────────────────────────────────────────────────────────────────────────────
# int8 helpers — replicate C int8_t wrap and the kernel's int16 clamp
# ─────────────────────────────────────────────────────────────────────────────
def i8(x):
    """Truncate a Python int to a signed 8-bit value (C int8_t cast)."""
    x &= 0xFF
    return x - 256 if x >= 128 else x


def clamp_i8(x):
    """Kernel saturation: sum>127 -> 127, sum<-128 -> -128."""
    if x > 127:
        return 127
    if x < -128:
        return -128
    return x


# ─────────────────────────────────────────────────────────────────────────────
# 1. Build the padded input DDR buffer  [INPUT_H_PAD, INPUT_W_PAD, INPUT_C_ALIGN]
#
#    Same DDR layout as simpleconv2d.cc (real pixels at (h+PAD, w+PAD), the PAD
#    border and the 4th channel stay 0), but the VALUES are now small zero-mean
#    random int8 samples in [-4, 4] instead of the saturating ramp (real_idx+1).
#    This keeps the accumulator small so the output tile is ~normally distributed.
# ─────────────────────────────────────────────────────────────────────────────
def build_input():
    buf = [0] * (INPUT_H_PAD * INPUT_W_PAD * INPUT_C_ALIGN)
    for h in range(INPUT_H):
        for w in range(INPUT_W):
            for c in range(INPUT_C):
                real_idx = (h * INPUT_W + w) * INPUT_C + c
                pos = ((h + PAD) * INPUT_W_PAD + (w + PAD)) * INPUT_C_ALIGN + c
                buf[pos] = mix32(real_idx + INPUT_SEED) % 9 - 4   # [-4, 4]
    return buf


# ─────────────────────────────────────────────────────────────────────────────
# 2. Build the filter in B^T [N, K] layout  (matches kernel B_ptr[f*K + kk])
#
#    Same B^T [N, K] layout as simpleconv2d.cc (the padding channel c == 3 weight
#    stays 0), but the real-channel weights are now zero-mean random samples from
#    {-1, 0, 1} instead of a constant 1. Mixing signs lets taps partially cancel
#    so the accumulator centers on 0 (normal distribution) rather than saturating.
# ─────────────────────────────────────────────────────────────────────────────
def build_filter():
    filt = [0] * (NUM_FILTERS * K)
    for f in range(NUM_FILTERS):
        for kh in range(KERNEL_H):
            for kw in range(KERNEL_W):
                for c in range(INPUT_C):
                    kk = (kh * KERNEL_W + kw) * INPUT_C_ALIGN + c
                    filt_key = ((f * KERNEL_H + kh) * KERNEL_W + kw) * INPUT_C + c
                    filt[f * K + kk] = mix32(filt_key + FILTER_SEED) % 3 - 1   # {-1,0,1}
    return filt


# ─────────────────────────────────────────────────────────────────────────────
# 3. Extract the FIRST slab the shim 2D BD delivers to core (0,0), round 0.
#
#    Outer height slice = rows[0:61] (mesh row 0), on-core round 0 = rows[0:19].
#    Width chunk 0        = cols[0:61]. All 4 channels.
#    The slab is a NARROW [TILE_H, TILE_W*C] = [19, 244] block cut from the padded
#    DDR buffer whose row pitch is INPUT_W_PAD*C = 230*4 = 920.
#    Returned flat, exactly as the kernel indexes it: slab[ih*raw_wc + iw*C + c].
# ─────────────────────────────────────────────────────────────────────────────
def extract_slab(input_buf, h0=0, w0=0):
    slab = [0] * (TILE_H * raw_wc)
    ddr_row_pitch = INPUT_W_PAD * INPUT_C_ALIGN   # 920
    for r in range(TILE_H):                       # 19 rows
        ddr_base = (h0 + r) * ddr_row_pitch + w0 * INPUT_C_ALIGN
        for col in range(raw_wc):                 # 244 = 61*4 contiguous bytes
            slab[r * raw_wc + col] = input_buf[ddr_base + col]
    return slab


# ─────────────────────────────────────────────────────────────────────────────
# 4. Kernel replay: on-chip im2col + matmul for one slab -> one output tile.
#
#    Exactly conv2d_spatial Phase 2:
#      for oh in [0,oh_per_row): for ow in [0,ow_dim): for j in [0,tile_cols):
#        sum = Σ_{kh,kw,c} slab[ih*raw_wc + iw*c_dim + c] * B_local[j*k_dim + kk]
#        (ih=oh*S+kh, iw=ow*S+kw, kk running 0..K-1)
#      clamp to int8, store local_out[(oh*ow_dim+ow)*tile_cols + j]
#
#    `trace` optionally records, for one chosen (oh,ow,j), every (input,filter)
#    tap so we can print all inputs/filters used for that output element.
# ─────────────────────────────────────────────────────────────────────────────
def compute_tile(slab, filt, trace_ohowj=None):
    out = [0] * (oh_per_row * ow_dim * tile_cols)
    trace = []
    for oh in range(oh_per_row):
        for ow in range(ow_dim):
            for j in range(tile_cols):
                s = 0
                kk = 0
                for kh in range(SP_KH):
                    for kw in range(SP_KW):
                        for c in range(SP_C):
                            ih = oh * SP_S + kh
                            iw = ow * SP_S + kw
                            iv = slab[ih * raw_wc + iw * SP_C + c]
                            fv = filt[j * k_dim + kk]
                            s += iv * fv
                            if trace_ohowj == (oh, ow, j):
                                trace.append((kh, kw, c, ih, iw, iv, fv, iv * fv))
                            kk += 1
                out[(oh * ow_dim + ow) * tile_cols + j] = clamp_i8(s)
    return out, trace


# ─────────────────────────────────────────────────────────────────────────────
# 5. Independent reference: naive padded conv2d for the same output region
#    (scalar_conv2d from simpleconv2d.h, restricted to the first tile) — a
#    cross-check that the slab-based kernel replay is correct.
# ─────────────────────────────────────────────────────────────────────────────
def reference_tile(input_buf, filt):
    ref = [0] * (oh_per_row * ow_dim * tile_cols)
    for oh in range(oh_per_row):
        for ow in range(ow_dim):
            for f in range(tile_cols):
                acc = 0
                for kh in range(KERNEL_H):
                    for kw in range(KERNEL_W):
                        for c in range(INPUT_C_ALIGN):
                            ih = oh * STRIDE + kh
                            iw = ow * STRIDE + kw
                            kk = (kh * KERNEL_W + kw) * INPUT_C_ALIGN + c
                            iv = input_buf[(ih * INPUT_W_PAD + iw) * INPUT_C_ALIGN + c]
                            fv = filt[f * K + kk]
                            acc += iv * fv
                ref[(oh * ow_dim + ow) * tile_cols + f] = clamp_i8(acc)
    return ref


# ─────────────────────────────────────────────────────────────────────────────
# 6. FULL convolution: the complete [OH=112, OW=112, OC=64] output over the whole
#    padded image and all 64 filters (naive padded conv2d = scalar_conv2d from
#    simpleconv2d.h). Byte-for-byte identical to what the hardware produces (the
#    per-tile kernel replay cross-checks against this same math).
#
#    Only real channels (c in [0, INPUT_C)) are summed: the pad channel c==3 has
#    a zero filter weight AND a zero input, so it contributes nothing. Zero filter
#    taps are pre-skipped so the ~118M-op scan stays tractable in pure Python.
# ─────────────────────────────────────────────────────────────────────────────
def full_conv(input_buf, filt):
    # Pre-collect the nonzero (kh, kw, c, weight) taps for every filter.
    taps = []
    for f in range(OUT_C):
        t = []
        for kh in range(KERNEL_H):
            for kw in range(KERNEL_W):
                for c in range(INPUT_C):          # pad channel weight is 0
                    kk = (kh * KERNEL_W + kw) * INPUT_C_ALIGN + c
                    w = filt[f * K + kk]
                    if w != 0:
                        t.append((kh, kw, c, w))
        taps.append(t)

    out = [0] * (OUT_H * OUT_W * OUT_C)
    for oh in range(OUT_H):
        ih0 = oh * STRIDE
        for ow in range(OUT_W):
            iw0 = ow * STRIDE
            obase = (oh * OUT_W + ow) * OUT_C
            for f in range(OUT_C):
                acc = 0
                for (kh, kw, c, w) in taps[f]:
                    pos = ((ih0 + kh) * INPUT_W_PAD + (iw0 + kw)) * INPUT_C_ALIGN + c
                    acc += input_buf[pos] * w
                out[obase + f] = clamp_i8(acc)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Pretty printers
# ─────────────────────────────────────────────────────────────────────────────
def print_slab(slab):
    print(f"\n=== INPUT SLAB used by core(0,0) round 0  [H={TILE_H}, W={TILE_W}, C={SP_C}] ===")
    print(f"    (slab flat layout: slab[ih*{raw_wc} + iw*{SP_C} + c], row pitch raw_wc={raw_wc})")
    for ih in range(TILE_H):
        print(f"  ih={ih:2d}:")
        for iw in range(TILE_W):
            vals = [slab[ih * raw_wc + iw * SP_C + c] for c in range(SP_C)]
            print(f"      iw={iw:2d}  c[0..3]= {vals}")


def print_filters(filt):
    print(f"\n=== FILTERS used: first {OC_PER_G} of {NUM_FILTERS}, each K={K} (7x7x4, B^T [N,K]) ===")
    print("    (real channels c0..c2 = 1, pad channel c3 = 0)")
    for f in range(OC_PER_G):
        print(f"  filter f={f:2d}:")
        for kh in range(KERNEL_H):
            row = []
            for kw in range(KERNEL_W):
                kk = (kh * KERNEL_W + kw) * INPUT_C_ALIGN
                cvals = [filt[f * K + kk + c] for c in range(INPUT_C_ALIGN)]
                row.append(cvals)
            print(f"      kh={kh}: " + "  ".join(str(cv) for cv in row))


def print_output(out, ref):
    print(f"\n=== OUTPUT TILE  [oh={oh_per_row}, ow={ow_dim}, f={tile_cols}] ===")
    print("    kernel replay value | (ref) — MISMATCH flagged if differ")
    mism = 0
    for oh in range(oh_per_row):
        for ow in range(ow_dim):
            line = []
            for f in range(tile_cols):
                idx = (oh * ow_dim + ow) * tile_cols + f
                k = out[idx]
                r = ref[idx]
                tag = "" if k == r else "  <<MISMATCH"
                if k != r:
                    mism += 1
                line.append(f"{k:4d}{('' if k==r else '/%d!' % r)}")
            print(f"  out[oh={oh},ow={ow:2d}] f0..15 = {line}")
    print(f"\n  output vs reference mismatches: {mism}")
    return mism


def print_full_output(out):
    """Print every value of the full [OH=112, OW=112, OC=64] output tensor.

    One line per (oh, ow) spatial position listing all OUT_C=64 filter values,
    so the whole 64*112*112 = 802,816-element output is emitted.
    """
    print(f"\n=== FULL OUTPUT  [oh={OUT_H}, ow={OUT_W}, f={OUT_C}]  "
          f"({OUT_C}*{OUT_H}*{OUT_W} = {OUT_C * OUT_H * OUT_W} values) ===")
    for oh in range(OUT_H):
        for ow in range(OUT_W):
            base = (oh * OUT_W + ow) * OUT_C
            vals = [out[base + f] for f in range(OUT_C)]
            print(f"  out[oh={oh:3d},ow={ow:3d}] f0..{OUT_C - 1} = {vals}")


def print_trace(trace, oh, ow, j, result):
    print(f"\n=== TAP TRACE for output[oh={oh}, ow={ow}, f={j}] ===")
    print("    every (kh,kw,c) -> input value, filter value, product")
    running = 0
    for (kh, kw, c, ih, iw, iv, fv, prod) in trace:
        running += prod
        print(f"    kh={kh} kw={kw} c={c}  ih={ih:2d} iw={iw:2d}  "
              f"in={iv:4d}  filt={fv}  prod={prod:5d}  acc={running:6d}")
    print(f"    raw acc = {running}  ->  clamped int8 = {result}")


# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("Conv2d spatial-halo DEBUG — FULL output (all filters, whole image)")
    print(f"  padded input : [{INPUT_H_PAD}, {INPUT_W_PAD}, {INPUT_C_ALIGN}]  (real at (h+{PAD},w+{PAD}))")
    print(f"  filters      : all {NUM_FILTERS}, K={K}")
    print(f"  full output  : [{OUT_H}, {OUT_W}, {OUT_C}]  "
          f"({OUT_C}*{OUT_H}*{OUT_W} = {OUT_C * OUT_H * OUT_W} values)")

    input_buf = build_input()
    filt      = build_filter()

    out = full_conv(input_buf, filt)
    print_full_output(out)


if __name__ == "__main__":
    main()
