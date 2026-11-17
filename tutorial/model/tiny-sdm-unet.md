# 用 aiegraph 把 tiny SD 的 UNet 搬上 AIE，生成一个橘子

本教程手把手带你把 **bk-sdm-tiny**（256×256 的 Stable Diffusion 小模型）的 UNet 部署到 VEK385 一代的板子（palmyra，A78 APU + AIE2PS）上：A78 跑 TVM 生成的 C 代码，conv、Linear、attention 跑在 AIE 上；然后让文本编码器和 VAE 在 host 上跑、UNet 的每一步都在板上跑，生成一张橘子图。

所有命令都在 palmyra213（10.23.224.213）上实际跑通过，结果见文末的[实测结果](#实测结果)。

![板上生成的橘子](tiny-sdm-unet-orange.png)

---

## 0. 它是怎么工作的

```
                    ┌────────────────────── build_unet.py（x86，一次，约 26 分钟）──────────────────────┐
 unet_sim.onnx ──► aiegraph.load ──► aiegraph.convert(target="a78") ──► TVM C：graph_driver.c（A78 上逐层调用）
                                         │
              partition_p4a.json ──► 156 层交给 AIE：48 conv + 72 Linear + 36 attention
                                         │
                  unet_aie.py / fa_attn.py：按层形状生成 aiehlc kernel（gemm_kacc_*、flash-attention）
                                         + 主机函数 unet_aie_conv / linear / attn（打包 → kernel<<<mesh>>> → 散布）
                                         │ 改写 graph_driver.c：这 156 个调用点改为调用主机函数
                                         ▼
                  aiegraph build ──► main.elf（650 MB：A78 代码 + AIE kernel + bf16 权重 + .aiehlc_io 输入段）
                    └──────────────────────────────────────────────────────────────────────────────────────┘

 出图（run_orange.sh）：
   prompt ─► CLIP 文本编码器（host）─► 25 步 × 2（CFG）= 50 次 UNet 调用 ─► VAE 解码（host）─► PNG
                                          │ 每次调用：
                                          │   aiehlc_elfio.py patch  把这次的输入写进 ELF 的 .aiehlc_io 段
                                          │   apppaltest.py --netload 把 ELF 送上板、跑一次 UNet
                                          │   从板子的输出转储读回结果，同时和 host 上的 onnxruntime 比 cos
```

几个关键点：

- **A78 上跑的是整网的 C 代码。** `aiegraph.convert` 用 TVM 把 ONNX 编成 C，`arm_build.py` 把它摊平成一个 `graph_driver.c`，按顺序逐层调用。板上的 A78 是 bare-metal，不需要 Linux 和 python。
- **AIE kernel 用 aiehlc 的编程模型写。** 每个 kernel 是一个 `__global__` 函数，用 `aie::program_id()`、`aie::load_buffer()`、`aie::store_buffer()` 描述每个 core 取哪块数据、写哪块；DMA、路由、锁都由 aiehlc 推导。主机端用 `device.partition(...)` 划出 4×4 共 16 个 core，再 `kernel<<<mesh>>>(...)` 启动。
- **一层一次启动。** conv 先在 A78 上做 im2col 变成矩阵乘，再交给 `gemm_kacc`：K 方向在 core 内累加，整层只启动一次。attention 的 QKᵀ、softmax、×V 融合成一个 flash-attention kernel。
- **运行时输入。** 输入放在 ELF 里一个专门的 `.aiehlc_io` 段，换输入不用重新构建，只要改这个段。
- **启动加速。** 已开：warm launch（kernel 只加载一次）、control packet AOT 的 5 个 pragma，以及 KCO（`#pragma KERNELCONFIGOFFLOAD`，core 自己配 DMA）。

---

## 1. 准备

**硬件**
- 一块装好"边收边放" netboot supervisor 的 palmyra 板，例如 palmyra213（10.23.224.213）。整网 ELF 有 650 MB，旧 supervisor 的 94 MiB staging 放不下。
- 这台机器能通过 `script/test/apppaltest.py --netload` 访问到这块板。

**软件**

| 需要什么 | 本教程用的路径 | 说明 |
|---|---|---|
| aiehlc 源码，`unet_tutorial` 分支 | `aiehlchj` 仓库 | 本教程的所有代码都在这个分支上 |
| LLVM/MLIR 构建 | `/scratch/staff/huaj/llvm-project/build` | 编 aiehlc 用 |
| Vitis 2026.x 和 aarch64 bare-metal 工具链 | 由 `script/setup.sh` 找到 | 和平常用 aiehlc 一样 |
| Python 环境（numpy、onnx、onnxruntime、Pillow，以及 aiegraph 的依赖） | `/scratch/staff/huaj/aiehlc/.venv/bin/python3` | 构建和出图都用它 |
| bk-sdm-tiny 的 ONNX | `/scratch/staff/huaj/modelzoo/artifacts/bk-sdm-tiny/`（`unet_sim.onnx`、`unet.onnx`、`pipeline_ref_256/p00_steps.npz`、`p1/parts/`） | 构建用 |
| bk-sdm-tiny 的模型库 | `/scratch/staff/huaj/aiehlc/model_zoo/bk-sdm-tiny/`（文本编码器、UNet、VAE、tokenizer） | 出图用 |
| `aie_imagegen`（含 `palmyra_stage_runner`） | `/scratch/staff/huaj/aiehlc/aiehlc-sdxl-wt/palmyra-runner/script` | aiehlc-sdxl 仓库，`jane/palmyra-stage-runner` 分支，至少要 7245e1f（含读回修复） |

下面用两个变量：

```bash
export AIEHLC=/scratch/staff/huaj/aiehlc/aiehlcopensource/aiehlchj   # 仓库根目录
export WORK=$HOME/unet_work                                          # 你的工作目录（放输入、构建产物和图）
export PY=/scratch/staff/huaj/aiehlc/.venv/bin/python3
mkdir -p $WORK
```

---

## 2. 取代码

用 worktree 检出 `unet_tutorial` 分支：

```bash
cd $AIEHLC
git worktree add .worktrees/unet_tutorial unet_tutorial
export T=$AIEHLC/.worktrees/unet_tutorial
```

worktree 里没有 BSP 的 workspace 和私有的 alib，要从一个已经装好的树链接过来（完整克隆的仓库跳过这一步）：

```bash
bash $T/script/worktree.sh link-deps $T --from $AIEHLC/.worktrees/unet_m2_int6 --private-alib
ls -d $T/thirdparty/arch/*/workspace      # 应该能看到 workspace
```

本教程要用到的文件都在 `$T/example/aiegraph/unet_bk_sdm_tiny/` 下：

| 文件 | 作用 |
|---|---|
| `build_unet.py` | 构建入口：ONNX → aiegraph → TVM C + AIE kernel → `main.elf` |
| `unet_aie.py` | 生成 `gemm_kacc_{ch}x{k}x{px}` kernel 和主机函数，改写 `graph_driver.c` |
| `fa_attn.py` | 生成 flash-attention kernel |
| `keep_layout_rt.py` | 层与层之间保持布局，去掉 A78 上的 layout 转换 |
| `partition_p4a.json` | 哪些层上 AIE |
| `a78_layer_cflags.txt` | A78 上 TVM C 代码的编译选项（NEON） |
| `export_real_io.py` | 导出一份真实的 UNet 输入和 x86 参考输出 |
| `run_orange.sh` | 出图：host 跑文本编码器和 VAE，板上跑 UNet |

---

## 3. 编 aiehlc

```bash
cmake -S $T -B $T/build \
      -DLLVM_DIR=/scratch/staff/huaj/llvm-project/build/lib/cmake/llvm \
      -DMLIR_DIR=/scratch/staff/huaj/llvm-project/build/lib/cmake/mlir
cmake --build $T/build --target aiehlc -j 32          # 约 1 分钟
ls -la $T/build/aiehlc
```

一定要用这个分支自己编出来的 aiehlc。它包含两个 KCO 的修复，用别的 aiehlc 打开 KCO 会链接失败或在板上卡死：
- `kernel_tm.h` 的链接修复（d5ee489）；
- 多 kernel 程序里 KCO 配置计划没有按 kernel 重置的修复（6776083）。

---

## 4. 导出一份真实输入

构建时要用一份真实输入，作为 `.aiehlc_io` 段的初始值，同时用来在板上自检：

```bash
/scratch/staff/huaj/modelzoo/venv/bin/python -I $T/example/aiegraph/unet_bk_sdm_tiny/export_real_io.py \
    --step 0 --branch cond --out $WORK/real_io
```

最后一行应该是：

```
REAL_IO OK parts=25 data=p00/step0/cond batch_idx=1 t=999 sample_crc32=150c323f unet_in_crc32=a5b911e2 ... sim_vs_full cos=1.0000000
```

`unet_in_crc32=a5b911e2` 是这份输入的指纹，后面板上的打印会用它核对。

---

## 5. 构建 UNet 的 ELF

```bash
cd $T/example/aiegraph/unet_bk_sdm_tiny
env -u AIEHLC_ROOT AIEHLC_BIN=$T/build/aiehlc $PY build_unet.py \
    --onnx /scratch/staff/huaj/modelzoo/artifacts/bk-sdm-tiny/unet_sim.onnx \
    --real-io $WORK/real_io \
    --wdtype bf16 \
    --partition partition_p4a.json \
    --aie conv,linear,attn --attn-fa --keep-layout --kincore \
    --launch-opt warm --ctrlpkt aot-kco \
    --arm-layer-cflags-file a78_layer_cflags.txt \
    --runtime-io --dump-out \
    --out $WORK/build 2>&1 | tee $WORK/build.log
```

每个开关的作用：

| 开关 | 作用 |
|---|---|
| `--wdtype bf16` | 权重存成 bf16（0.65 GB）；AIE 上 bf16 乘法、fp32 累加 |
| `--partition partition_p4a.json` | 哪些层上 AIE |
| `--aie conv,linear,attn` | conv、Linear、attention 这三类上 AIE |
| `--attn-fa` | attention 的 QKᵀ、softmax、×V 融合成一个 flash-attention kernel |
| `--keep-layout` | AIE 的输出直接写成下一层要的布局，去掉 A78 上 56 个 layout 转换 |
| `--kincore` | 一层一次启动，K 方向在 core 内累加（不加这个开关，就会退回到每个 64³ 小块启动一次） |
| `--launch-opt warm` | kernel 只加载一次，之后反复启动 |
| `--ctrlpkt aot-kco` | control packet AOT 的 5 个 pragma，再加上 KCO |
| `--arm-layer-cflags-file` | A78 上 TVM C 代码的 NEON 编译选项 |
| `--runtime-io --dump-out` | 输入放进 `.aiehlc_io` 段，输出转储到控制台，出图时要用 |

约 26 分钟。日志里应该能看到：

```
[build_unet] KINCORE layers=156 launches_planned=258 kernel=bf16_gemm_kacc; shapes=38
[build_unet] ctrlpkt=aot pragmas=6
[build_unet] graph_driver.c: 156 partition call sites -> unet_aie_conv (48) + unet_aie_linear (72) + unet_aie_attn (36)
[build_unet] AIO section ... bytes=253248 n_in=3 crc32=a5b911e2 dump_out=1
[build_unet] built .../main.elf -> $WORK/build/main.elf (657,579,872 B; weights.bin 646,757,376 B)
GATE_RC=0
```

想看生成出来的东西：
- 改写后的图驱动：`$WORK/build/ag_work/arm_build/graph_driver.c`（找 `unet_aie_conv(`）；
- 所有 AIE kernel 和主机函数：`$WORK/build/aiek_src/aiek_unetconv.cc`。

---

## 6. 在板上跑一次 UNet（冒烟测试）

先用构建时自带的那份输入跑一次，确认 ELF 在板上是对的：

```bash
cd $T
$PY script/test/apppaltest.py -board palmyra -host 10.23.224.213 -y --netload \
    --netload-wait 1800 --netload-reply-timeout 600 --netload-max-run-ms 300000 \
    $WORK/build/main.elf 2>&1 | tee $WORK/smoke.log | \
    grep -E "^Host: |\[netboot\] run|in_crc32_board|^unet: graph_run|^LAUNCH warm|^CMP unet"
```

应该看到（约 35 秒）：

```
[devlock] acquired 10.23.224.213 (...)
Host: huaj@10.23.224.213
[netboot] run 711: placed 655298632 bytes, ... stream=1 elf_size=657579872
INPUT unet sample+timestep_i32+ehs in_crc32_board=a5b911e2 expect=a5b911e2 MATCH (timestep=999) runtime_io=1
unet: graph_run 1295.470 ms, output elems=4096 (ref 4096)
LAUNCH warm ms_per_launch=2.937 launches=258 aie_wall_ms=757.633 ... ctrlpkt=aot aot_fallbacks=0
CMP unet board=palmyra213 data=p00/step0/cond in_crc32=a5b911e2 cos=1.000000 maxabs=0.00324592 dtype=bf16 aie_launches=258 coreen_calls=258 run_rc=0 aie_flop_frac=1.000 PASS
```

怎么看这几行：
- `in_crc32_board=… MATCH`：板子读到的输入就是你给的那份；
- `aie_launches=258 coreen_calls=258`：AIE 真的被启动了 258 次，没有悄悄退回到 A78 上算；
- `cos=1.000000 … PASS`：和 x86 上 onnxruntime 的结果一致；
- `graph_run 1295.470 ms`：一次 UNet 调用的时间。

**注意：** `apppaltest.py` 就算板上打印了 FAIL 也会返回 0，所以一定要看 `CMP … PASS` 这一行，不要只看返回码。

---

## 7. 生成橘子

```bash
cd $T/example/aiegraph/unet_bk_sdm_tiny
./run_orange.sh $WORK/build/main.elf $WORK/img 2 25      # seed 2，25 步：50 次 UNet 调用都在板上
```

`run_orange.sh` 做了三件事：
1. 在 `$WORK/img/zoo/bk-sdm-tiny/` 建一个私有模型库，链接到 bk-sdm-tiny 的 ONNX 文件，再写一个 `aie/aie_manifest.json`，把 UNet 这一步交给板上的 runner。runner 的命令行大致是：

   ```
   python -m aie_imagegen.palmyra_stage_runner {in} {out} \
       --elf $WORK/build/main.elf --elfio $T/src/tool/elfio/aiehlc_elfio.py --launcher $T \
       --host 10.23.224.213 --calls-dir ... --steps-log ... \
       --map timestep=timestep_i32:int32 --map encoder_hidden_states=ehs --out-shape 1,4,32,32 \
       --ref-onnx .../bk-sdm-tiny/unet.onnx
   ```

   两个 `--map` 把 `aie_imagegen` 给出的输入名（`timestep` 是 int64、`encoder_hidden_states`）对应到 ELF 里的输入槽（`timestep_i32`、`ehs`）。
2. 运行 `aie_imagegen --backend aie`：文本编码器和 VAE 在 host 上；每次 UNet 调用，runner 把这次的输入写进 ELF 的副本，netload 到板上跑，再读回结果。没有 host 回退：任何一次板上调用失败，这张图就失败。
3. 汇总：每次调用写一行 `STEP …` 到 `$WORK/img/steps_s2.log`，里面有板上读到的输入 crc、`graph_run_ms`，以及和 onnxruntime 比的 cos。

参数：提示词 "a photo of an orange, citrus fruit, high quality"，负提示词 "blurry, deformed, low quality"，256×256，guidance 7.5。可以用环境变量 `PROMPT`、`NEG`、`GUIDANCE`、`BOARD` 改。

**和 host 版本对比（可选）。** 先在 host 上用同样的参数生成一张参考图：

```bash
cd $WORK && PYTHONPATH=/scratch/staff/huaj/aiehlc/aiehlc-sdxl-wt/palmyra-runner/script $PY -m aie_imagegen \
    -p "a photo of an orange, citrus fruit, high quality" --negative-prompt "blurry, deformed, low quality" \
    -o $WORK/host_seed2.png --backend host --zoo /scratch/staff/huaj/aiehlc/model_zoo \
    --model bk-sdm-tiny --size 256 --steps 25 --seed 2 --guidance 7.5 --decoder vae
```

再让 `run_orange.sh` 顺便算 PSNR 和 cos：

```bash
REF=$WORK/host_seed2.png ./run_orange.sh $WORK/build/main.elf $WORK/img 2 25
```

---

## 实测结果

2026-11-16 在 palmyra213 上实测，seed 2，25 步：

```
[run_orange] UNet calls on the board: 50, cos_OK: 50, mean graph_run_ms: 1308.8
[run_orange] vs host_seed2.png: PSNR 20.11 dB, cos 0.97028
```

| 指标 | 数值 |
|---|---|
| UNet 调用 | 50 次（25 步 × CFG 2），全部在板上，没有回退，`aot_fallbacks=0` |
| 每次调用的板上输入 crc | 50/50 MATCH |
| 单次调用和 onnxruntime 比的 cos | 最低 0.999972，大多数是 0.999999 |
| 单次 UNet 的 `graph_run` | 1295–1320 ms，平均 1308.8 ms（其中 AIE 258 次启动 ≈ 758 ms，每次 2.94 ms） |
| 每次调用的墙钟时间 | 31–39 s，绝大部分是 650 MB ELF 的 netload，不是计算 |
| 整张图 | 1822 s：UNet 1814 s，文本编码器 0.23 s，VAE 0.75 s（这两个在 host 上） |

左边是板上生成的，右边是 host 上 onnxruntime（fp32）用同样参数生成的：

![板上 vs host](tiny-sdm-unet-orange-vs-host.png)

两张图的构图和颜色是一样的。整图 PSNR 只有 20 dB，是因为 AIE 上的权重是 bf16，每一步的误差（单步 cos ≈ 0.99999）会沿着 25 步的去噪过程累积放大，所以细节（纹理、背景条纹）会有偏移。单次调用 cos ≥ 0.99997 才是判断"板上算对了"的标准，整图的 PSNR 只用来看有没有跑偏。

**墙钟时间为什么是 30 分钟，而不是 50 × 1.3 s ≈ 65 s。** 现在每次 UNet 调用都要把整个 ELF（含 0.65 GB 权重）重新 netload 到板上，只为了换 250 KB 的输入。要把一张图降到一分钟左右，需要让 ELF 常驻在板上，每步只传输入和输出（这也是下一步要做的事）。

---

## 常见问题

| 现象 | 原因和办法 |
|---|---|
| `apppaltest` 一直不动，打印 `[devlock] waiting` | 板子正被别人用（比如团队的 sprint）。它会自动排队，`--netload-wait` 内拿到锁就会继续 |
| `elf_size … exceeds staging 98566144` | 板上还是旧的 netboot supervisor，换成"边收边放"的版本 |
| 构建时 KCO 的 kernel 链接失败（`__aiehlc_tm_anchor is undefined`） | 用的不是本分支编出来的 aiehlc。把 `AIEHLC_BIN` 指向 `$T/build/aiehlc` |
| 打开 KCO 后，板上卡在第 4 次 launch（`wait_io TIMEOUT`） | 同上，旧的 aiehlc 没有 6776083 的修复 |
| 构建报 `thirdparty/alib` 相关的错，或两个构建互相弄坏 | 两个构建共用了同一个 alib。worktree 里用 `link-deps … --private-alib`，或者不要同时构建 |
| 结果行里夹着 `[DEBUG]` 打印，正则匹配不上 | `apppaltest` 的调试输出，先用 `grep -v DEBUG` 去掉；它也可能插在一行的中间（见下面 `readout_crc_ok=0` 那条） |
| 图是糊的 | 步数太少（4 步）或者提示词不好；用 25 步和上面的提示词 |
| `STEP k=0 rc=4`，报告里是 `readout: 16384 B of output (need 236544 B)` | runner 不知道输出形状，默认拿了第一个输入（ehs）的形状。加 `--out-shape 1,4,32,32`（`run_orange.sh` 已经加了） |
| 前几步都好，某一步 `rc=4 … readout_crc_ok=0`，`board_crc32` 却是 MATCH 的 | 板上算对了，是读回时坏了：host 的 `[.. DEBUG] main: program_done detected…` 打印插进了某一行 `AIO_D` 的中间，把这行十六进制拆成两半，转储的 crc 就对不上。用带修复的 runner（`RE_DEBUG` 能去掉行中间的 DEBUG 片段，再把被拆开的行接回去） |
| `STEP … cos_OK=0` | 看那次调用的 `calls_s2/call_<k>/console.log`：输入 crc 是否 MATCH、是否 `CMP … PASS` |

---

## 下一步：改哪里

| 想做什么 | 改哪里 |
|---|---|
| 换一组层上 AIE | `partition_p4a.json`，或者用 `src/aiegraph/autopart.py` 重新生成 |
| 改 gemm_kacc kernel 的写法或切法 | `unet_aie.py` 的 `KACC_KERNEL`（kernel 模板）和 `kacc_geom()`（切法） |
| 改 flash-attention | `fa_attn.py` 的 `FA_KERNEL` 和 `FA_HOST` |
| 单独测一个 kernel | `kacc_bench.sh --gemm` / `--bmm` |
| 看时间花在哪 | 板上日志里的 `LAUNCH split` 和 `PROF` 行 |

现在第 5 步里，除了载入、转换和链接，offload 是 `build_unet.py` 在生成好的 `graph_driver.c` 上用正则改写的，并没有完全走 aiegraph。改成纯 aiegraph 调用的方案见 [`docs/plans/2026-11-16-aiegraph-group-offload-design.md`](../../docs/plans/2026-11-16-aiegraph-group-offload-design.md)。

现在还没做完的：softmax 以外的 Norm（GroupNorm、LayerNorm）还在 A78 上（`--aie …,norm` 在 `unet_m3_int` 分支上），conv 的 im2col 还在 A78 上，层与层之间的数据都经过 DDR，每次启动都是同步的，只用了 144 个 tile 里的 16 个。
