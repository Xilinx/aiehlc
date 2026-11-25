<!-- Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
     SPDX-License-Identifier: Apache-2.0 -->
# Proposal: tune a model build offline with the cost model

Status legend: **[implemented]** = code exists and runs today (cited); **[prototype]** = `tune_loop.py` in this
directory, runs offline on archived data; **[proposed]** = not built.

## Goal

Board time is the scarce resource (one palmyra213 run of the three nets takes minutes per cfg, the
17-kernel K calibration ~7 min plus 1338 s of builds — M7 REPORT §4). Use `estimate()` to discard bad build
candidates *before* the board, send only the best few, and feed the board results back into the parameters.

```
 knobs ──► 1 enumerate candidates ──► 2 compile ──► 3 estimate() each ──► 4 rank, keep top-k / within margin
                                                                                  │
          6 refit params from those runs ◄── 5 board runs (top-k only) ◄──────────┘
                     │
                     └──► frozen params/ (md5-pinned) ──► next round of 3
```

## Step by step

| # | step | status | where |
|---|---|---|---|
| 1 | Enumerate candidates over knobs: control-path cfg (off / aot / aot-kco) | **[implemented]** for cfg | build scripts take the cfg (`script/perf/build_net_cfg.sh`, `build_unet_cfg.sh`); `tune_loop.py` `enumerate_candidates` finds the 6 builds on disk |
| 1b | …over tiling knobs (kacc `S`, `kch`, `nkc`, `ntile` via `aiegraph.cnn_chunk.kacc_geom`), partition (which layers/sites on the AIE), fusion | **[proposed]** | the DES already takes any kacc geometry (`des.py:74-83`), but see limit L1 |
| 2 | Compile each candidate (only the build, no board) | **[implemented]** (builds) | ~1338 s for 17 kernel ELFs (M7 REPORT §4); fast-build work reduces this |
| 3 | `estimate()` each candidate, keep `unpriced` | **[implemented]** | `estimate.py:444`; net mode needs only the build dir |
| 4 | Rank per net; board set = top-k ∪ {within `margin` of best} | **[prototype]** | `tune_loop.py` `select`; margin should be ≥ the model's per-total error (§Limits) |
| 5 | Board runs of the selected candidates only | **[prototype: replay]** | `tune_loop.py` reads the archived M7 logs instead of running the board; the real loop would call `script/perf/p0_one_kernel.sh` / the net run scripts |
| 6 | Refit from the new board logs, re-freeze | **[implemented]** for K (kernel level) and L/N/site-K tools; **[prototype]** wiring | `tools/costmodel_fit.py` `calibrate`, `tools/l_fit.py`, `tools/n_fit.py`, `tools/k_net.py`; `tune_loop.py` `refit` refits K on the 17 M7 kernel logs and writes only to `--refit-out` |
| 7 | Guard: a refit must reproduce or beat the frozen params on held-out runs before it replaces them (LOO, as `costmodel_fit.loo_maxrel`) | **[implemented]** as a tool, **[proposed]** as an automatic gate | `tools/costmodel_fit.py` `loo_maxrel` |

## Prototype run (real output, this worktree, no board)

```
$ /scratch/staff/huaj/aiehlc/.venv/bin/python3 -I src/costmodel/example/offline_tuning_proposal/tune_loop.py
CAND net=resnet18 cfg=off     pred_ms=    588.9 sites=21 unpriced=-
CAND net=resnet18 cfg=aot-kco pred_ms=    423.7 sites=21 unpriced=-
CAND net=unet     cfg=off     pred_ms=   1553.7 sites=138 unpriced=-
CAND net=unet     cfg=aot-kco pred_ms=    777.8 sites=138 unpriced=-
CAND net=yolo26n  cfg=off     pred_ms=   9629.0 sites=102 unpriced=-
CAND net=yolo26n  cfg=aot     pred_ms=   8647.9 sites=102 unpriced=-
RANK net=resnet18 order=aot-kco:423.7>off:588.9 board=aot-kco pick=aot-kco(meas 422.6 ms) board_best=aot-kco(meas 422.6 ms) kept
RANK net=unet     order=aot-kco:777.8>off:1553.7 board=aot-kco pick=aot-kco(meas 796.9 ms) board_best=aot-kco(meas 796.9 ms) kept
RANK net=yolo26n  order=aot:8647.9>off:9629.0 board=aot pick=aot(meas 8739.5 ms) board_best=aot(meas 8739.5 ms) kept
REFIT b_prefetch_b     frozen=164140.0965  refit=163840.0
REFIT chunk_ovh_cyc    frozen=58.42        refit=58.64
REFIT compute_exposed  frozen=0.0          refit=0.0
REFIT ddr_bw_scale     frozen=1.7503       refit=1.6616
REFIT ddr_model        frozen=bytes        refit=bytes
REFIT ddr_run0_b       frozen=51.1319      refit=46.27
REFIT k_form           frozen=m5           refit=m5
REFIT round_fixed_cyc  frozen=12048.6471   refit=11589.4909
REFIT wait_call_cyc    frozen=582.2456     refit=617.3177
REFIT kernels=17 logs=/scratch/staff/huaj/tmp/costmodel_m7/kernels/palmyra213 maxrel frozen=0.253 refit=0.247 (in-sample)
TUNE nets=3 candidates=6 board_runs=3 saved=3 best_kept=3/3 refit_maxrel=0.247 frozen_maxrel=0.253
```

Reading it: 3 of 6 board runs avoided and the board-best cfg kept for all 3 nets (the same decision as M6
`costmodel_cfg_decision.py`). `frozen_maxrel=0.253` reproduces M7 REPORT §3b. The refit's in-sample gain
(0.253 → 0.247) is small and **in-sample**; step 7 (LOO) must decide whether it replaces the frozen file.
Runtime ~19 s on x86.

## Limits — be honest

* **L1 Post-compilation only** (COSTMODEL.md ch. 9 "what is not done yet" #1). Every input of `estimate()` comes from a
  compiled build: static features from host.cc / aiek sources, core cycles from ISS or `kernel.lst`
  (`tools/kcyc_lst.py`, `tools/kcyc_unet.py`). So step 2 (compile) cannot be skipped and the search space is
  bounded by build time, not by `estimate()` (which takes milliseconds to seconds). Enumerating tiling knobs
  without compiling needs a core-cycle model for unbuilt kacc instances — **[proposed]**, not available; do not
  feed invented `core_cyc` to the DES.
* **L2 Unpriced paths count as 0** (#3). Op/bdgraph mode lists `resident`, `aot_window` / other BD write paths,
  locks, `glue`, `l` in `unpriced` and leaves them out of `total_ms` (`estimate.py:119-171`, `:480`). A ranking
  across candidates that differ in an unpriced term is meaningless; the loop must refuse such comparisons
  (`tune_loop.py` prints `unpriced=` per candidate; net mode returns `[]`).
* **L3 One board** (#4). Full parameter sets exist only for palmyra213 (`estimate.py:186`, `:344`); VEK385 has a
  kernel-level diff only (M7.2.2). The loop as written tunes for palmyra213.
* **L4 Resolution** (#2, #5). The candidates ranked so far differ 1.1–2×. Per-site maxrel is 0.177 (UNet) and
  per-site board jitter ~2–4 %; ranking candidates a few percent apart is unvalidated — hence the `--margin`
  rule (send all candidates within the model error to the board).
* **L5 The replay is not a board run.** Stage 5 of the prototype reads archived logs; it proves the plumbing and
  the decision, not new measurements.
* **L6 Refit scope.** The prototype refits the kernel-level K calibration only (`palmyra213_kcal.m5.json`). Net
  mode uses the site-level frozen sets (`k_*`, `l_*`, `n_*`); their refit tools exist (`tools/k_net.py --out`,
  `tools/l_fit.py`, `tools/n_fit.py --fit`) but are not wired into the loop — **[proposed]**.
