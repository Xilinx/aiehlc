# Cost-model examples (offline, x86, no board)

All commands run from the repository root with the project venv; outputs below are the real outputs of this
worktree (2026-11-23). No example writes to `params/` or touches a board.

```bash
PY=/scratch/staff/huaj/aiehlc/.venv/bin/python3
```

Data each example reads:

| example | in the repository | outside (archived on /scratch, read only) |
|---|---|---|
| A single_kernel | board log `src/costmodel/test/data/costmodel/u_lin_1024x320x320.log`, `params/` | — (the ISS row is inlined from `aie2ps_kernel_cycles.json`) |
| B whole_model | `params/`, M6 golden TSVs in `src/costmodel/test/data/costmodel/` | build dirs `/scratch/staff/huaj/tmp/costmodel_m40` (resnet18, yolo26n), `/scratch/staff/huaj/tmp/costmodel_m31223` (unet); M34/M41/M42 board logs; M7 `pred_vs_meas.tsv` (cross-check only) |
| C offline_tuning_proposal | `params/` | the same build dirs + logs as B; the 17 M7 kernel logs `/scratch/staff/huaj/tmp/costmodel_m7/kernels/palmyra213`; `estimate.CYCLES` |

The build dirs and board logs are the ones `example/aiegraph/costmodel_cfg_decision.py` and
`tools/report_scatter.py` (the M6 / M7 gates) already read; they are too large to commit.

## A. `single_kernel/` — one kernel, predicted vs measured

```
$ $PY -I src/costmodel/example/single_kernel/estimate_kernel.py
BOARD machine=aarch64 midr=0x410fd423 board=palmyra213
measured  rt_wait_io = 0.5166 ms/launch  (launches=8, src/costmodel/test/data/costmodel/u_lin_1024x320x320.log)
predicted estimate() default host (5.0 us/BD)      = 0.4716 ms  rel=-0.087  bound=ddr  unpriced=glue,host_bd_us(default 5.0),l
predicted estimate() host bd_us=5.203 from log     = 0.4716 ms  rel=-0.087  bound=ddr  unpriced=glue,l
predicted des.predict_launch, M5 cal (kcal.m5)     = 0.5250 ms  rel=+0.016  bound=ddr
DES terms (M5): compute_ms=0.7128 ddr_ms=0.6275 dma_ms=0.2091 stall_ms=0.2096 rounds=16
EXAMPLE_A kernel=u_lin_1024x320x320 meas_ms=0.5166 pred_m5_ms=0.5250 rel=+0.016 PASS
```

What it shows:
* Segment **K only** (the AIE-side `rt_wait_io`); `estimate()` lists `glue` (N) and `l` (L) as unpriced — they
  are not in `total_ms`.
* `estimate()` uses the default K calibration (`params/palmyra213_cal.json`, K2c form); the M5 form
  (`params/palmyra213_kcal.m5.json`, the one behind M7 REPORT §3b) is closer here. Its 0.5250 ms equals the M7
  report's prediction for this kernel; the report's measured 0.5180 is a later board run, the committed log
  gives 0.5166.
* The host BD cost from the log (5.203 µs/BD) does not move this prediction: with this calibration
  `overlap_bd = 0` (`des.py:150`) and the b-prefetch cap binds at `b_prefetch_b` (`des.py:142`).
* `bound=ddr`: the 16 rounds are DMA/DDR-paced (`des.py:183`), consistent with M7 §6 ("17 kernels all ddr bound").

## B. `whole_model/` — a whole network, per site + total

```
$ $PY -I src/costmodel/example/whole_model/estimate_net.py            # --net resnet18 (default)
== resnet18 cfg=off board=palmyra213  build=/scratch/staff/huaj/tmp/costmodel_m40/resnet18_off
   log=/scratch/staff/huaj/aiengineers/shuhua/clones/cost/tasks/2026-11-23_M41/resnet18_off.d.log
   site       launches   T_hat_ms    pred_ms    meas_ms     rel
   part1            49     1.9763    96.8387    97.8840  -0.011
   part3            13     3.6863    47.9219    47.2520  +0.014
   part4            13     3.5761    46.4893    46.4810  +0.000
   part5            13     3.5761    46.4893    46.1990  +0.006
   part6            13     3.5761    46.4893    46.4980  -0.000
   part19            1    31.0807    31.0807    31.3560  -0.009
   part20            1    29.6175    29.6175    29.5560  +0.002
   part21            1    29.6175    29.6175    28.9170  +0.024
   part9             4     6.1153    24.4612    24.2210  +0.010
   part10            4     5.7450    22.9800    22.4270  +0.025
   part11            4     5.7450    22.9800    22.6880  +0.013
   part14            4     5.5487    22.1948    22.3250  -0.006
   part15            4     5.1829    20.7316    20.6690  +0.003
   part16            4     5.1829    20.7316    20.7940  -0.003
   part17            1    17.4949    17.4949    17.7430  -0.014
   part7             4     4.0348    16.1392    16.1570  -0.001
   part12            4     3.7790    15.1160    15.3120  -0.013
   part24            1     9.8698     9.8698    10.2810  -0.040
   part13            4     2.0643     8.2572     8.3880  -0.016
   part8             4     1.9899     7.9596     7.7350  +0.029
   part18            1     5.4213     5.4213     5.5110  -0.016
   TOTAL sites=21 pred_ms=588.9 meas_ms=588.4 rel=+0.001 (estimate total_ms=588.8814)
   check M6 golden T_hat: identical; M7 pred_vs_meas.tsv rows: identical
== resnet18 cfg=aot-kco board=palmyra213  build=/scratch/staff/huaj/tmp/costmodel_m40/resnet18_aot-kco
   log=/scratch/staff/huaj/aiengineers/shuhua/clones/cost/tasks/2026-11-23_M41/resnet18_aot-kco.d.log
   site       launches   T_hat_ms    pred_ms    meas_ms     rel
   part1            49     1.0403    50.9747    50.7900  +0.004
   part3            13     2.6720    34.7360    34.2550  +0.014
   ...                                     (21 rows, see the run)
   part18            1     2.2928     2.2928     2.3560  -0.027
   TOTAL sites=21 pred_ms=423.7 meas_ms=422.6 rel=+0.003 (estimate total_ms=423.7065)
   check M6 golden T_hat: identical; M7 pred_vs_meas.tsv rows: identical
EXAMPLE_B net=resnet18 pts=42 r2=1.000 maxrel=0.040 PASS
```

UNet (138 sites per cfg; `--rows 8` prints the 8 largest predicted sites):

```
$ $PY -I src/costmodel/example/whole_model/estimate_net.py --net unet --rows 8
== unet cfg=off board=palmyra213  build=/scratch/staff/huaj/tmp/costmodel_m31223/unet_off
   log=/scratch/staff/huaj/aiengineers/shuhua/clones/cost/tasks/2026-11-22_M34/unet_off.a.log
   site       launches   T_hat_ms    pred_ms    meas_ms     rel
   conv121           1    89.0554    89.0554    82.0650  +0.085
   conv122           1    35.7298    35.7298    36.5910  -0.024
   linear49          1    34.7758    34.7758    33.1490  +0.049
   linear66          1    34.7758    34.7758    33.1370  +0.049
   linear83          1    34.7758    34.7758    33.1570  +0.049
   linear13          2    15.2488    30.4976    27.9210  +0.092
   linear136         2    15.2488    30.4976    27.7620  +0.099
   linear153         2    15.2488    30.4976    27.9320  +0.092
   ... 130 more sites (--rows to show)
   TOTAL sites=138 pred_ms=1553.7 meas_ms=1524.8 rel=+0.019 (estimate total_ms=1553.6594)
   check M6 golden T_hat: n/a; M7 pred_vs_meas.tsv rows: identical
== unet cfg=aot-kco board=palmyra213  build=/scratch/staff/huaj/tmp/costmodel_m31223/unet_aot-kco
   log=/scratch/staff/huaj/aiengineers/shuhua/clones/cost/tasks/2026-11-22_M34/unet_aot-kco.a.log
   site       launches   T_hat_ms    pred_ms    meas_ms     rel
   conv121           1    72.9093    72.9093    80.6500  -0.096
   conv122           1    30.0401    30.0401    32.7470  -0.083
   conv86            1    22.6232    22.6232    23.7670  -0.048
   conv52            1    20.7470    20.7470    21.6890  -0.043
   conv139           1    20.7182    20.7182    23.0520  -0.101
   linear13          2     9.5028    19.0056    17.3480  +0.096
   linear136         2     9.5028    19.0056    17.2670  +0.101
   linear153         2     9.5028    19.0056    17.4230  +0.091
   ... 130 more sites (--rows to show)
   TOTAL sites=138 pred_ms=777.8 meas_ms=796.9 rel=-0.024 (estimate total_ms=777.8330)
   check M6 golden T_hat: n/a; M7 pred_vs_meas.tsv rows: identical
EXAMPLE_B net=unet pts=276 r2=0.991 maxrel=0.177 PASS
```

What it shows: prediction from the build dir only (`estimate()` net mode, frozen params, nothing refit);
measured site_ms from the archived board log; each row is identical to the M7 report's `pred_vs_meas.tsv`, and
the resnet18 `T_hat` is identical to the committed M6 golden. The UNet per-site maxrel 0.177 is the thin margin
noted in the package README §6.

## C. `offline_tuning_proposal/` — using the model to tune a build offline

Design: [`offline_tuning_proposal/PROPOSAL.md`](offline_tuning_proposal/PROPOSAL.md) (implemented vs
proposed, limits). Prototype run and its real output are in that file:

```
$ $PY -I src/costmodel/example/offline_tuning_proposal/tune_loop.py
...
TUNE nets=3 candidates=6 board_runs=3 saved=3 best_kept=3/3 refit_maxrel=0.247 frozen_maxrel=0.253
```

Options: `--top-k N`, `--margin X` (also send candidates predicted within X of the best), `--no-refit`,
`--refit-out FILE` (write the refit K calibration there; `params/` is never written).
