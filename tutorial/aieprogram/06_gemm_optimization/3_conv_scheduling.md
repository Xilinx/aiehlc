# 6.3 The same scheduling options on the conv2d: `{.depth}`, `{.resident}`

Example: [`example/tileprogram/ccode/simpleconv2dkernel_port_opts.cc`](../../../example/tileprogram/ccode/simpleconv2dkernel_port_opts.cc)
-- [1b](../01_contraction/conv2d.md)'s `simpleconv2dkernel_port_less.cc` with
**three changed lines**.

## The whole program

The whole program. The only difference from 1b is the last argument of the three
`load_buffer` / `store_buffer` calls (`{.resident = true}`, `{.depth = 2}`).

<!-- code: example/tileprogram/ccode/simpleconv2dkernel_port_opts.cc -->
```cpp
#include "simpleconv2d_new_load.h"
// simpleconv2dkernel_port_less.cc plus per-port schedule options (DMA side only, same result):
//   {.resident = true} weights: one block per core, loaded once, held in ONE buffer all launch
//   {.depth = 2}       input slab / output block: ping-pong, the DMA fills one while the core uses the other
__global__ void conv2d_spatial(aie::port<input_window_int8 *, aie::IFM, aie::shape<230, 230, 4>> win_a,
                               aie::port<input_window_int8 *, aie::WEIGHTS, aie::shape<64, 7, 7, 4>> win_b,
                               aie::port<output_window_int8 *, aie::OFM, aie::shape<112, 112, 64>> win_c) {
    const int KH = 7, KW = 7, S = 2, C = 4;         // 7x7 stride-2 conv, 4 (aligned) channels
    const int oh_t = 7, ow_t = 28, oc_t = 16;       // output block computed per round
    const int ih_t = (oh_t - 1) * S + KH;           // 19: input footprint of one output block
    const int iw_t = (ow_t - 1) * S + KW;           // 61
    const int oh_core = 112 / aie::num_programs(0); // 28 output rows per mesh row
    const int pid_h = aie::program_id(0);           // mesh row: which OH band
    const int pid_c = aie::program_id(1);           // mesh col: which OC group
    const int8_t *w = aie::load_buffer(win_b, {pid_c * oc_t, 0, 0, 0}, {oc_t, KH, KW, C}, {.resident = true});
    for (int i = 0; i < oh_core / oh_t; i++) {          // 4 row blocks
        for (int j = 0; j < 112 / ow_t; j++) {          // 4 column blocks
            const int oh0 = pid_h * oh_core + i * oh_t; // output block origin
            const int ow0 = j * ow_t;
            const int8_t *x = aie::load_buffer(win_a, {oh0 * S, ow0 * S, 0}, {ih_t, iw_t, C}, {.depth = 2});
            int8_t *y = aie::store_buffer(win_c, {oh0, ow0, pid_c * oc_t}, {oh_t, ow_t, oc_t}, {.depth = 2});
            for (int oh = 0; oh < oh_t; oh++)
                for (int ow = 0; ow < ow_t; ow++)
                    for (int oc = 0; oc < oc_t; oc++) {
                        int acc = 0;
                        for (int kh = 0; kh < KH; kh++)
                            for (int kw = 0; kw < KW; kw++)
                                for (int c = 0; c < C; c++)
                                    acc += x[((oh * S + kh) * iw_t + ow * S + kw) * C + c] *
                                           w[((oc * KH + kh) * KW + kw) * C + c];
                        y[(oh * ow_t + ow) * oc_t + oc] = (int8_t)(acc > 127 ? 127 : acc < -128 ? -128 : acc);
                    }
            aie::release(win_a);
            aie::commit(win_c);
        }
    }
    aie::release(win_b);
}
int main() {
    aieSetDevice(0);
    aieArray device;
    aieMesh mesh = device.partition({3, 6, 0, 6}, HW_ROWS, HW_COLS);
    const int in_n = INPUT_H_PAD * INPUT_W_PAD * INPUT_C_ALIGN, w_n = K * NUM_FILTERS, out_n = M * N;
    int8_t *input = (int8_t *)__Runtime_Alloc(in_n), *filter = (int8_t *)__Runtime_Alloc(w_n);
    int8_t *output = (int8_t *)__Runtime_Alloc(out_n), *ref = (int8_t *)__Runtime_Alloc(out_n);
    memset(output, 0, out_n);
    init_conv2d_test_data(input, filter); // simpleconv2d_new_load.h
    XTime t0, t1;
    XTime_GetTime(&t0);
    conv2d_spatial<<<mesh>>>(input, filter, output, M, N, K);
    XTime_GetTime(&t1);
    printf("[BENCH] aie_wall_ms=%.3f\n", 1.0 * (t1 - t0) / COUNTS_PER_SECOND * 1000.0);
    scalar_conv2d(input, filter, ref);
    int bad = 0;
    for (int i = 0; i < out_n; i++)
        bad += output[i] != ref[i];
    printf("%s: %d mismatches out of %d\n", bad ? "FAIL" : "PASS", bad, out_n);
    return 0;
}
```
<!-- /code -->

So far the kernel said *what* each core computes and *which blocks* it uses.
It did not say *how* the blocks move: how many buffers a port gets on the
core, and whether a block is sent once or with every round. The compiler
picked defaults. This tutorial shows how to choose, per port, without touching
the compute.

## 1. Two separate questions: compute and data movement

Every port is a stream of blocks between DDR and a core. For each port the
schedule has to answer:

1. **How many core buffers?** With one buffer, the DMA can only refill it after
   the core `release`s it, so the core waits for every transfer. With two
   (ping-pong), the DMA fills one while the core works on the other.
2. **How often is the block sent?** A block the core needs for the whole
   launch (the filters) should be sent once and kept, in one buffer -- not
   double-buffered, not re-sent.

![What {.depth} and {.resident} change](../img/sched_timeline.svg)

The defaults: every port double-buffered (if the buffers fit), each block
streamed when its round comes. Options override that **per port**.

## 2. The options

They are the optional last argument of `load_buffer` / `store_buffer`, a
designated initializer:

```cpp
aie::load_buffer(port, offset, shape, {.depth = 2});
aie::load_buffer(port, offset, shape, {.resident = true});
aie::load_buffer(port, offset, shape, {.depth = 2, .tile = {8, 8}});   // 2-D blocks only
aie::store_buffer(port, offset, shape, {.depth = 2});                  // stores take only .depth
```

| Option | Meaning | Allowed when |
|---|---|---|
| `.depth = 1` / `2` | this port's core buffers: single / ping-pong | 1 or 2; at most that many blocks of the port held at once |
| `.resident = true` | one block per core, **sent once**, kept in **one** buffer for the whole launch | the kernel takes exactly one block of the port per core; not with `.depth = 2` |
| `.tile = {tr, tc}` | loads of 2-D blocks: the DMA delivers the block panel-major -- `tc`-wide column panels one after another -- so every `tr x tc` tile is contiguous (what a matrix unit wants) | divides the block; tile row and tensor row a multiple of 4 bytes |

Initializers go in the order `.reuse, .pad, .depth, .resident, .tile`, and
**every block of a port must pass the same options** -- they configure the
port's DMA, one setting per port. On the core they disappear: the compiler
drops the argument from the kernel code, so the compute is byte-for-byte the
same.

## 3. The change, line by line

The three lines that differ from `simpleconv2dkernel_port_less.cc`:

```cpp
    const int8_t *w = aie::load_buffer(win_b, {pid_c * oc_t, 0, 0, 0}, {oc_t, KH, KW, C}, {.resident = true});
```
The filters are one block per core, used in all 16 rounds: **resident**. One
3136-byte buffer instead of a ping-pong pair, sent once.

```cpp
            const int8_t *x = aie::load_buffer(win_a, {oh0 * S, ow0 * S, 0}, {ih_t, iw_t, C}, {.depth = 2});
            int8_t *y = aie::store_buffer(win_c, {oh0, ow0, pid_c * oc_t}, {oh_t, ow_t, oc_t}, {.depth = 2});
```
The input slab and the output block: **ping-pong**, so slab `k + 1` arrives
while block `k` is computed, and block `k - 1` drains while block `k` is
written. On this conv that is also what the compiler would choose (the halo
path is always ping-pong); writing it down makes the schedule explicit and
checked.

Everything else -- the loops, the MACs, the host -- is unchanged.

## 4. What changed in the build

The kernel analysis prints the options with each port:

```
win_a (IFM, ...): block (19,61,4), ... rounds 4x4 steps (14,0,0) / (0,56,0), options {.depth = 2}
win_b (WEIGHTS, ...): block (16,7,7,4), ... rounds 1x1 ..., options {.resident = true}
win_c (OFM, ...): block (7,28,16), ... rounds 4x4 steps (7,0,0) / (0,28,0), options {.depth = 2}
```

and the pipeline confirms it follows the resident port:

```
[KernelBlock] flowIdx=1 input follows the kernel's blocks: 1 x 3136 B per core, round stride 0/0 B
```

The generated code, `_less` vs `_opts`:

| | `simpleconv2dkernel_port_less.cc` | `simpleconv2dkernel_port_opts.cc` |
|---|---|---|
| weight window in `kernel.cc` | `window_init(..., buf_in_ping_1, ..., buf_in_pong_1, ...)` | `window_init(..., buf_in_ping_1, ..., buf_in_ping_1, ...)` -- one buffer |
| core buffers (`aieml.bcf`) | weights 0x78000 + 0x78c40, slab 0x79880 + 0x7aaa0, output 0x7bcc0 + 0x7c900 | weights 0x78000, slab 0x78c40 + 0x79e60, output 0x7b080 + 0x7bcc0 |
| buffer memory per core | 21,816 B | 18,680 B |
| weight S2MM BDs per core (`host.cc`) | 2, chained ping -> pong | 1 |
| `routing.cc` | -- | identical |

Same data movement on the wire (the weights were already one transfer per
mesh column), one buffer and one buffer descriptor less per core.

## 5. Measured

```
simpleconv2dkernel_port_less.cc   [BENCH] aie_wall_ms=93.845   PASS: 0 mismatches out of 802816
simpleconv2dkernel_port_opts.cc   [BENCH] aie_wall_ms=85.855   PASS: 0 mismatches out of 802816
```

8 ms (9 %) from one option, with identical compute -- because a launch here is
dominated by **configuration**, not by the MACs. The port buffers are part of
the kernel's data image, which every launch writes into all 16 cores over the
configuration bus at roughly 0.4 us per 32-bit word
(`doc/performance/register_write_cost.md`). One 3136-byte buffer less on 16
cores is 12,544 words, about 4.7 ms by that rate, plus the buffer descriptor
and lock writes that went with it. Memory freed on the core is also memory a
bigger tile can use.

## 6. The rules, as the compiler enforces them

Try each on `simpleconv2dkernel_port_opts.cc` with
`./build/aiehlc-convtile trace <file.cc>`:

| Change | Result |
|---|---|
| slab `{.depth = 1}` | `ERROR: ... win_a (IFM): {.depth = 1} is not supported on a spatial-halo (conv) kernel yet -- its window lowering is ping-pong; use {.depth = 2}, or {.resident = true} for a block taken once` |
| slab `{.resident = true}` | `ERROR: ... {.resident = true}, but the kernel takes 16 blocks of this port per core; a resident block is loaded once (outside the loop) and held for the whole launch` |
| slab `{.depth = 2, .tile = {1, 4}}` | `ERROR: ... {.tile} re-lays out a 2-D block; this port's blocks have 3 dims` |
| slab `{.depth = j == 0 ? 1 : 2}` | `ERROR: ... block 1 on core (0,0) passes different {.depth / .resident / .tile} options than block 0 (they configure the port's DMA, one setting per port)` |

Options never change what is computed, so a wrong option is a build error,
never a wrong answer.

## 7. Where the other options pay off: the GEMM path

On a GEMM-style kernel (`fullconnect_auto`, no halo -- 1a) the
defaults are less suited than on this conv, and the options do more:

- the GEMM schedule **re-sends B with every M round**; `{.resident = true}` on
  B sends it once and keeps it;
- `{.depth}` sets one port's buffering on its own (the example ping-pongs its
  A strips while B is resident; `{.depth = 1}` would single-buffer a port that
  must make room for others);
- `{.tile = {8, 8}}` makes the DMA deliver A in 8 x 8 tiles, the operand layout
  of `aie::mmul<8,8,8>`, so the core never re-arranges A.

`example/tileprogram/ccode/simplegemm_dataflow.cc` uses all three
(256 x 256 x 256 x 8 panels per launch, GOLDEN PASS 524,288 / 524,288,
79.6 ms on the board). Design notes: `doc/design/trace_to_routing.md`,
section "Dataflow options".

Next: [6.4 Vector instructions + scheduling on the conv2d](4_conv_vector_scheduling.md).
