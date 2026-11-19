# bk-sdm-tiny UNet on the Palmyra APU + AIE

One UNet step of bk-sdm-tiny (256×256 image, latent 4×32×32, timestep, text
states 77×768) built with aiegraph → TVM C for the A78, with the conv layers in
`partition.json` running on the AIE. Sprint `unet-bf16-streamload`, 2026-11-13.

## Files
| File | What |
|---|---|
| `build_unet.py` | Build entry: ONNX → aiegraph.convert (a78, fp32 graph, bf16 weights) → `main.elf` with the 3 real inputs embedded and a cos/maxabs check against x86 ORT |
| `unet_aie.py` | `--aie conv`: maps each partition layer (ONNX Conv) to its TVM call, builds the AIE archive on one carrier layer, patches `graph_driver.c` so those call sites call `aiednn_aie_conv()` |
| `partition.json` | The 37 conv layers on the AIE, 86,600 planned launches, aie_flop_frac 0.417 |
| `p5_board.log` | The board run (P5) these files produced |

No C text lives in this directory (except `build_unet.py`'s `MAIN_C`): every kernel and host-glue
template is read from the aiednn library, the board compare tool from the aiegraph runtime:

| Template | What |
|---|---|
| [`src/aiednn/conv2d/`](../../../src/aiednn/conv2d/) | `bf16_gemm_convin.cc` (`gemm_bf16`, one 64×64×64 bf16 tile per launch), `kacc_conv_host*.cc` (im2col / scatter glue of the K-in-core conv), `tile64_conv_*.cc`, `conv_bf16_carrier_ref.c` |
| [`src/aiednn/gemm/`](../../../src/aiednn/gemm/) | `bf16_gemm_kacc.cc`, `gemm_kacc_kernel.cc` (one instance per layer shape), `kacc_linear_host.cc`, control-packet pragmas, `kacc_bench_*` |
| [`src/aiednn/attention/`](../../../src/aiednn/attention/) | flash attention (`fa_*.cc`), the bmm host glue `kacc_bmm_host.cc` |
| [`src/aiednn/norm/`](../../../src/aiednn/norm/) | `fa_online_softmax.cc` (online softmax + 1/l rescale) |
| [`src/aiednn/elementwise/`](../../../src/aiednn/elementwise/) | `epilogue_bias_residual.cc` (bias / temb / residual adds) |
| [`src/aiednn/layout/`](../../../src/aiednn/layout/) | `neon_tr4_f32.cc` (NEON 4×4 fp32 transpose) |
| [`src/aiegraph/runtime/cmp_cos.c`](../../../src/aiegraph/runtime/cmp_cos.c) | `p1_cmp()`: prints the `CMP <model> … cos=… PASS` line |

Lookup: `aiednn.template("<category>/<file>")` (verbatim text), `aiednn.params(...)` (its `@NAME@`
parameters), `aiednn.render(..., NAME=value)`.

## Build and run
```sh
AGPY=<aiehlc venv python>
$AGPY build_unet.py --onnx <bk-sdm-tiny>/unet_sim.onnx --real-io <real_io> \
    --wdtype bf16 --aie conv --partition partition.json --launch-opt warm --out out_p5
python3 script/test/apppaltest.py -board palmyra -host <board> -y --netload \
    --netload-reply-timeout 600 --netload-max-run-ms 2400000 out_p5/main.elf
```
`<real_io>` holds `sample.bin`, `timestep_i32.bin`, `ehs.bin`, `unet_out.bin`
(x86 ORT reference) for one step/branch.

Needs from this branch:
- `--launch-opt warm`: runtime `AIEHLC_ZFILL_WIDE`, `AIEHLC_PMLOAD_WIDE`,
  `AIEHLC_PM_WARM_SKIP`, `aiehlc.sh --skip-bss` (kernel loaded once, 44.4 → 4.5 ms per launch).
- the 650 MB ELF: netboot stream-place (`src/tool/netboot`), the ELF no longer
  has to fit the 94 MiB staging area.
- `AIEGRAPH_WDTYPE=bf16`, `AIEGRAPH_PARAMS_INPLACE=1`, multi-input graphs (aiegraph).

## Result (palmyra213, 2026-11-13)
- `cos=1.000000 maxabs=0.00272` vs x86 ORT, 86,600 AIE launches, `kloads=1`.
- One UNet call: 437.6 s with AIE (392 s of it launch overhead), 61 s A78-only.

Known limits: one tiny tile per launch and soft-float math make the AIE path
slower than the A78. Next: one launch per layer with K accumulated on the core,
bf16 vector MAC, then Linear layers.
