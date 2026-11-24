# `costmodel` — the AIE local cost model

A white-box model that predicts, **without a board**, how long one AIE launch (and so one site or a whole
network) takes on a given board, and says *where* the time goes (K, L or N). Calibrated parameters are
frozen per board in `params/`. Everything below cites the code (`file:line`, paths relative to this
directory unless noted). Background: the tutorial chapter 9 (`COSTMODEL.md`, "这个模型是什么，现在能做什么")
and the M7 report (`tasks/2026-11-23_M7/REPORT.md`) of the cost channel.

## 1. What the model is

### 1.1 Five-term decomposition: `T = K + L + N − overlap`

Every launch is split into three measured segments (board-log fields parsed by `launch_split.py`):

| term | meaning | board quantity | model |
|---|---|---|---|
| **K** | AIE side: the host time in `__Runtime_wait_io` that the host's own BD issue does not hide | `rt_wait_io` of `LAUNCH split_detail` (`des.py:3-6`) | `des.py` (kernel level), `tools/k_diag.py:1204` `khats_predict` (UNet sites), `tools/k_net.py:125` `k_predict` (ResNet18/yolo26n sites) |
| **L** | launch overhead `kload + core_reset + coreen + rt_bdcfg + rt_startio` | `LAUNCH split` / `split_detail` (`launch_l.py:3-5`) | linear in 7 static features per control-path cfg: `launch_l.py:27-28` (`CFGS`, `FEATURES`), `predict_l` `launch_l.py:81-89` |
| **N** | host glue (packing, scatter, flush/inval, BD-config glue) | `PROF aie_host_split` | per-class linear `N = Σ coef·f + d` from static build features: `n_feat.py`, `n_layout.py`, `ap_sites.py`, `cc_sites.py`, fitted by `tools/n_fit.py:163` `fit_kind` |
| **overlap** | the part of N (glue) that runs while the AIE is busy | — | `N_ovl = β·ovl`; UNet: `K_hat = max(0, body − N_ovl/launches) + t0 + kfirst/launches` (`estimate.py:239`, `tools/k_diag.py:1206-1208`); CNN nets: `K = max(core + dma, N_ovl) − N_ovl + c` (`tools/k_net.py:127-130`) |

Per site the model returns `T_hat = (K + L + N) / launches` ms per launch (`estimate.py:242-243`,
`tools/kln_site.py:109-111`), which is the same as `L + N_pre + max(busy, N_ovl) + N_post` when the AIE
is busier than the overlapping glue (`estimate.py:242`) — the "− overlap" in the title.

### 1.2 Roofline flag

`roofline.py:13-24`: intensity = MACs / mesh-unique DDR bytes per launch; `bound = "compute"` when the core
compute cycles exceed the DDR read cycles, else `"ddr"`. It is a label for reports (the K fit prints it per
kernel, `tools/costmodel_fit.py` `[K] ... roof=`); the time itself comes from the DES.

### 1.3 Discrete-event layer (`des.py`)

`des.predict_launch` (`des.py:121-184`) simulates one `gemm_kacc` launch on the 4×4 mesh: `R = ntile`
rounds, per round 24 host BDs (`des.py:63`), `nkc` K-chunks in 2-deep ping-pong windows; chunk *j*
arrives at `max(arrival j−1, release of chunk j−2) + dma_chunk` (`des.py:162-169`); DDR time per burst or
bytes-only (`des.py:90-119`); the host waits `max(0, round + fixed − overlap)` per round (`des.py:170-171`).
Returns `wait_ms` (= K) plus `compute_ms / dma_ms / lock_ms / stall_ms / ddr_ms / bound`.
The calibrated constants and their uncalibrated defaults are `CAL_DEFAULTS` (`des.py:49-61`).

### 1.4 Per-board calibrated, frozen parameters (`params/`)

| file | what | written by |
|---|---|---|
| `palmyra213.json` | microbenchmark physics (clock, DDR/DMA bandwidth, BD switch, lock, burst points) | `tools/mb_fit.py` via `script/perf/run_microbench.sh` |
| `palmyra213_cal.json`, `palmyra213_kcal*.json` | K (DES) calibration; `kcal.m5` = M5 form (`--form m5`) | `tools/costmodel_fit.py` |
| `l_palmyra213*.json` | L coefficients per cfg | `tools/l_fit.py`, `tools/m3_site.py --lrefit` |
| `n_*_palmyra213*.json` | N coefficients per net / class | `tools/n_fit.py` |
| `k_*_palmyra213*.json`, `kdw_*.json` | site-level K (UNet `khats`, CNN `theta`) | `tools/k_diag.py --khat-static`, `tools/k_net.py`, `tools/k_dmafeat.py` |
| `kcyc_*.json` | static per-launch core cycles | `tools/kcyc_unet.py`, `tools/kcore_net.py` |
| `m41_frozen.md5`, `m42_frozen.md5` | the frozen file set of resnet18 / yolo26n, md5-pinned | hand-frozen at M4.1 / M4.2 |

Which files `estimate()` loads is fixed in code: UNet `KLN_PARAMS` (`estimate.py:182-190`), CNN nets
`NET_PARAMS` → md5 file (`estimate.py:340-344`); every file is md5-checked when loaded
(`estimate.py:347-363`) and reported in the result's `"params"`. Only **palmyra213** has a full set;
`board_params()` raises for any other board (`estimate.py:59-64`).

## 2. Module map

| module | role |
|---|---|
| `estimate.py` | **the public entry** `estimate()` (`estimate.py:440`): op/group mode (`estimate_op` :98, `estimate_bdgraph` :132), UNet site mode (`estimate_sites` :227), net mode (`estimate_net` :366); `board_params` :59 |
| `des.py` | discrete-event K model of one kacc launch (§1.3) |
| `roofline.py` | compute- vs DDR-bound flag (§1.2) |
| `launch_l.py`, `launch_split.py` | L model; parser of the board's `LAUNCH split*` lines |
| `static_feat.py`, `kload_static.py`, `rst_static.py`, `kswitch.py` | static L features from a build dir (calls, words, kload words, core resets, kernel switches) |
| `n_feat.py`, `n_layout.py`, `unet_seq.py` | static N features / site table of a UNet build |
| `ap_sites.py`, `cc_sites.py` | static N features of resnet18 (ap_gemm) / yolo26n (kacc_chunk) builds |
| `bdgraph.py` | build-JSON (`dfscheduleprovenancemap`) form of `estimate()`: BD counts, issue time, DDR floor (`bdgraph.py:205`, `:447`) |
| `cli.py`, `__main__.py` | `python -m costmodel` (§3) |
| `tools/` | fitting / measurement / report tools (moved from `script/perf/`, §5) |
| `params/` | frozen per-board parameters (moved from `script/perf/params/`) |
| `example/` | three runnable examples (`example/README.md`) |
| `test/` | the cost-model tests and their data (`test/data/costmodel/`): `python3 -m pytest -q src/costmodel/test` |

## 3. Command line

The package imports nothing heavy (no TVM). From the repository root:

```bash
PY=/scratch/staff/huaj/aiehlc/.venv/bin/python3
cd src && $PY -m costmodel --help          # or: PYTHONPATH=src $PY -m costmodel ...
```

| subcommand | does |
|---|---|
| `estimate --net N --build ROOT --cfg C [--sites a,b] [--params NAME] [--json]` | net / site mode, frozen params |
| `estimate --kernel NAME [--cycles JSON] [--bd-us X] [--launches N]` | one kacc kernel (K only; L/N unpriced) |
| `estimate --group-json FILE` | any `estimate()` input (schema `estimate.py:1-40`) |
| `fit k\|l\|n\|mb\|scmb\|kcyc\|kcore\|knet\|kdiag ARGS` | runs the fitting tool (`python -I tools/<tool>.py ARGS`) |
| `report scatter\|params ARGS` | `tools/report_scatter.py` / `tools/params_table.py` |
| `tool NAME ARGS` / `tools` | any tool / list them |
| `params` | frozen files + md5 |
| `selftest` | offline: 9 tool selftests + the md5 pins of `params/*.md5` |

Real output (this worktree):

```
$ $PY -m costmodel selftest
COSTMODEL SELFTEST tool=k_noise rc=0 KNOISE SELFTEST PASS (4/4)
COSTMODEL SELFTEST tool=k_dmafeat rc=0 KDMAF SELFTEST PASS (27/27)
COSTMODEL SELFTEST tool=k_ovl_check rc=0 KOVL SELFTEST PASS (18/18)
COSTMODEL SELFTEST tool=kln_site rc=0 KLN SELFTEST PASS (17/17)
COSTMODEL SELFTEST tool=k_diag rc=0 KDIAG SELFTEST PASS (37/37)
COSTMODEL SELFTEST tool=t_noise rc=0 TNOISE SELFTEST PASS
COSTMODEL SELFTEST tool=net_sites rc=0 NETSITES_SELFTEST n=12 bad=0 PASS
COSTMODEL SELFTEST tool=tsv_cmp rc=0 TSVCMP SELFTEST PASS (12/12)
COSTMODEL SELFTEST tool=net_kdiag rc=0 NETKDIAG SELFTEST PASS (8/8)
COSTMODEL SELFTEST frozen_md5 files=2 bad=0
COSTMODEL SELFTEST tools=9 bad=0 PASS

$ $PY -m costmodel estimate --net unet --build /scratch/staff/huaj/tmp/costmodel_m31223 --cfg aot-kco --sites conv121,linear13
ESTIMATE net=unet cfg=aot-kco board=palmyra213 sites=2 total_ms=91.9149
model: K+L+N frozen (M34 PZ): T_hat = (K_hat x launches + L_hat + N_hat) / launches
params src/costmodel/params/k_palmyra213.m3122.json md5=3f9f2123fd09ee34100028c64fef1dc4
params src/costmodel/params/kcyc_unet_iss.json md5=599a25137259750b85a9431551525536
params src/costmodel/params/kdw_palmyra213.m3112c.json md5=786032f24141823c59c8abb5d61a984d
params src/costmodel/params/l_palmyra213.m3121.json md5=ade402695101ca3816935dd78c9800e9
params src/costmodel/params/n_unet_palmyra213.m3121.json md5=951402a1beafb38a698c2462d6129e8b
site=linear13   launches=2     T_hat_ms=9.5028 site_ms=19.0056
site=conv121    launches=1     T_hat_ms=72.9093 site_ms=72.9093

$ $PY -m costmodel estimate --kernel u_lin_1024x320x320
ESTIMATE board=palmyra213 launches=1 total_ms=0.4716 wait=0.4716 compute=0.7128 ddr=0.6609 l_ms=- issue_ms=- bound=ddr
unpriced (counted as 0 in total_ms): glue, host_bd_us(default 5.0), l
```

`--kernel` reads the ISS cycles table `estimate.CYCLES` (`estimate.py:54`, a file on /scratch, not in the
repo); pass `--cycles` for another one.

## 4. Step-by-step usage

### Step 1 — have a compiled build (the model is post-compilation)

Net/site mode reads the build directory only (host.cc, aiek sources, kernel listings), never a board log:
`<ROOT>/<net>_<cfg>/` from `example/aiegraph/<net>/build_*.py` or `script/perf/build_net_cfg.sh`
(UNet: `example/aiegraph/unet_bk_sdm_tiny/build_unet_cfg.sh`). Kernel mode needs the kernel's kacc
geometry + ISS `core_cyc` (`estimate.py:98-112`).

### Step 2 — estimate

```python
import sys; sys.path.insert(0, "src")
from costmodel import estimate
r = estimate({"net": "resnet18", "build": "/scratch/staff/huaj/tmp/costmodel_m40", "cfg": "off"})
r["total_ms"], r["sites"][0], r["params"], r["unpriced"]
```

Three input forms (`estimate.py:440-480`): `{"net", "build", "cfg"}` (net mode), `{"unet_build", "cfg"}`
(UNet site mode), or a list of ops / `{"ops": [...], "launches": n}` (op mode; K from the DES, L only if the
op carries `"l"` features, N never). **Always read `r["unpriced"]`:** op-mode `total_ms` is the sum of the
priced terms only (`estimate.py:476`); unpriced terms (`glue`, `l`, `resident`, BD write paths other than
the fitted one, locks …) are listed and count as 0 (`estimate.py:115-126`, `:137-167`).

### Step 3 — compare with the board (optional, offline once the log exists)

The measured side comes from the board log: per-site `PROF part=… site_ms=…` lines
(`example/aiegraph/costmodel_cfg_decision.py` `read_log`) or, per kernel, `LAUNCH split_detail …
rt_wait_io=…` (`tools/costmodel_fit.py` `load_point`). `report scatter` reproduces the M7 figure:

```
$ $PY -I script/perf/report_scatter.py --out $TMPDIR/scat      # old path, runs the moved tool
M7SCAT net=resnet18 pts=42 r2=1.000 maxrel=0.040
M7SCAT net=unet pts=276 r2=0.991 maxrel=0.177
M7SCAT net=yolo26n pts=204 r2=0.999 maxrel=0.104
M7SCAT nets=3 board=palmyra213 pts=522 png=1 r2_min=0.991 PASS
```

Its `pred_vs_meas.tsv` is byte-identical (`cmp`) to the M7 report's `scatter/pred_vs_meas.tsv`.

### Step 4 — (re)calibrate for a board

Board side (stays in `script/perf/`, needs a board): `run_microbench.sh --board <ip> --name <board>` →
`params/<board>.json`; `p0_one_kernel.sh` → per-kernel logs. Offline fits on those logs:

```bash
$PY -m costmodel fit k --stage K --form m5 --params src/costmodel/params/<board>.json \
    --cycles <aie2ps_kernel_cycles.json> --logs <kernel log dir> --out k.tsv --cal-out <board>_kcal.m5.json
$PY -m costmodel fit l --help    # L;   fit n --help: N;   fit knet --help: CNN site K
```

Then freeze: commit the json into `params/`, pin it (md5 file / `KLN_PARAMS`), and let
`script/perf/kln_frozen.sh` / `tools/m6_equiv.py` / `tools/m61_equiv.py` prove `estimate()` reproduces the
frozen chain bit for bit.

## 5. What moved, what stayed

The package first lived at `src/aiegraph/costmodel/` (import `aiegraph.costmodel`); it is now the
top-level package `src/costmodel/` (import `costmodel`, listed in `pyproject.toml`; re-run `pip install -e .`
in an existing venv so it is found without `PYTHONPATH=src`). Its tests and test data moved with it from
`src/aiegraph/test/` to `test/`. `aiegraph.aiperf.cost_model` imports `costmodel.estimate`. The build-input
keys of `build_net_cfg.sh` / `build_unet_cfg.sh` still exclude `src/aiegraph/costmodel`, so the commits that
touched the old location keep out of the key.

Moved with `git mv` (history kept), each old path left as a 13-line shim that `runpy`-runs the moved file as
`__main__`, or executes it in the importer's globals when loaded with `importlib` / bare `import`
(`script/perf/<tool>.py`):

* `script/perf/*.py` → `tools/` — 29 tools (`python -m costmodel tools` lists them).
* `script/perf/params/` → `params/`; `script/perf/params` is now a **symlink** to it, so old
  `--params script/perf/params/...` arguments in gates keep working. The md5 files list the new paths.

Stayed in `script/perf/`, on purpose:

| file | why |
|---|---|
| `run_microbench.sh`, `p0_one_kernel.sh`, `p0_kacc_host.cc`, `build_net_cfg.sh` | board / build side (SSH, `apppaltest.py`, aiehlc builds); `p0_one_kernel.sh` reads `p0_kacc_host.cc` from its own dir and its STAMP hashes that file, so moving would change every stamped build key. They now call the tools / write params at the new paths. |
| `kln_frozen.sh` | the frozen K+L+N gate chain, invoked by path from many gate scripts; it now calls `src/costmodel/tools/*.py` and reads `params/` (its `KLNFROZEN PARAMS file=` lines name the new paths). |
| `build_fresh.py`, `kgeom_equiv.py` | build-provenance tools (are the UNet build logs still valid for HEAD?), tied to `build_unet_cfg.sh` `INPUT_PATHS`, not cost-model code. |
| `win_lint.py` | C lint of timing windows in runtime instrumentation, not cost-model code. |

Runtime profiling hooks (`src/mlir/runtime`, `src/aiednn` host timers, `src/aiegraph/aiperf.py`) are
instrumentation and were not touched. The build-input keys (`INPUT_PATHS` of
`example/aiegraph/unet_bk_sdm_tiny/build_unet_cfg.sh:55` and `script/perf/build_net_cfg.sh:67`) include `src/aiegraph`
(where the kernel / host assembly lives since main's unet-decl refactor) but exclude `src/costmodel` and
`src/aiegraph/test`, so cost-model edits never mark a build stale.

## 6. Limits (from the code and COSTMODEL.md ch. 9 "还没做到的")

1. **Post-compilation only.** Inputs are a compiled build (static features) and ISS core cycles; there is no
   pre-compile estimate.
2. **Unpriced terms count as 0** in op-mode `total_ms` (`estimate.py:476`); `resident`, the `aot_window`
   BD path and locks are listed in `unpriced`, not priced.
3. **One board calibrated.** Full parameter sets exist only for palmyra213 (`KLN_PARAMS`, `NET_PARAMS`);
   VEK385 has a kernel-level comparison only (M7.2.2).
4. **Thin margin.** UNet per-site maxrel 0.177 against a 0.20 gate (M7 report 3a); per-site board jitter
   alone is ~2–4 %. K kernel fit: maxrel 0.253, LOO 0.295 (M7 report 3b).
5. **Used for cfg selection so far.** The candidates it has ranked (off vs aot-kco / aot) differ 1.1–2×; ranking
   of candidates a few percent apart is not validated (see `example/offline_tuning_proposal/PROPOSAL.md`).
