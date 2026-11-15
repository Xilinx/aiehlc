---
name: outputpingponghang
description: Kernel-only (aie::shape port) kernel builds and passes flowcheck but hangs in `wait_io TIMEOUT tile(c,0) ch=.. dir=0` (shim S2MM, output) when written WITHOUT dataflow options, while the same kernel with {.depth = 2} works; build log shows `pp_depth=1` for the output tensor; host.cc gives the core output MM2S ONE BD with next_bd=-1 while kernel.cc's output window_init alternates buf_out_ping_0 / buf_out_pong_0. Also: `{.depth = 1} on aie::store_buffer` analysis error.
---

# Output single-buffered on the host, ping-pong on the core -> hang

## Symptom
`simplegemm_opt0_baseline.cc` (2048x256x256, B block 16 KB, no options) built
rc=0, flowcheck OK, and hung on the board every run:

    [aie_runtime] wait_io TIMEOUT tile(3,0) ch=0 dir=0 pending=1

The same kernel with `{.resident}` / `{.depth = 2}` passed.

## Cause
Ports whose ping + pong do not fit the 29.7 KB buffer region drop to one
buffer (`routingHints`: `pp_depth = 1`). That depth reached the OUTPUT's
GemmSpace too, so the host lowered the core's output MM2S as ONE BD with
`next_bd = -1`. But `kernel.cc` always declares and alternates an output
ping AND pong (`window_init(window_out, ..., buf_out_ping_0, ..., buf_out_pong_0, ..., 256, 32)`;
`RoutingPort::declaredBytes` already counted both). The DMA sent block 0 from
ping, never re-armed, never read pong -> the shim S2MM waits forever.

## Fix (2026-11-15)
- `RoutingPort::ppDepth` (`trace_routing.h`): an output is always 2.
- `checkBlockOpts` (`kernel_trace.cpp`): `{.depth = 1}` on `store_buffer` is an
  analysis error ("an output window is always ping-pong on the core").
- Tests: `test_trace_routing` (`routingSpaces(bare, hb)[2].pp_depth == 2`),
  `kernel_trace_cases.py` "dataflow: .depth = 1 on a store".
Board: opt0 GOLDEN PASS 524,288 (90.6 ms, twice); simplegemm_kernel_port,
simplegemm_dataflow, opt1, conv _less, conv _vec_opts byte-identical outputs.

## Check
    grep -n "window_init(window_window_out" aout/worklocal/kernel.cc     # ping + pong
    grep "DMA BD Config" aout/worklocal/host.cc | grep "len=<out block>"  # must chain ping<->pong (next_bd set)

## Note when reading logs after a hang
The next run's console first drains the hung run's leftovers (a `wait_io
TIMEOUT` line before `[netboot] run N: placed ...`); judge a run only by the
lines after its own `run N` line.
