# 用 aiegraph 把 ResNet18 搬上 AIE：真实图片分类

本教程把 **ResNet18**（ImageNet，1×3×224×224 → 1000 logits）部署到 palmyra 板（A78 APU + AIE2PS）上：A78 跑 TVM 生成的整网 C 代码，**全部 20 个 Conv 和 1 个 Gemm** 都通过 aiegraph 的组 offload 跑在 AIE 上，然后在板上对真实图片分类，并和 onnxruntime 比对。

它和 [tiny-SD UNet](tiny-sdm-unet.md)、[YOLOv26](yolov26.md) 用的是同一套 aiegraph。区别是 CNN 不需要手写 partition：`aiegraph.cnn.offload_all()` 把每个 Conv / 带权重的 Gemm 自动匹配成一个组，放在 bf16 `gemm_kacc` 实例上。用户只写一个 `build_resnet18.py`（第 3 节贴出了全文），里面没有任何 C 文本。

所有数字都是在 palmyra213（10.23.224.213）上实测的，见[实测结果](#实测结果)。

---

## 0. 它是怎么工作的

```
resnet18_fp32_op17.onnx ─► ag.load ─► m.convert(target="a78") ─► TVM C：graph_driver.c（A78 上逐层调用）
                                         │
                       aiegraph.cnn.offload_all(emod, model, out)
                         │  plan_layers：每个 Conv / Gemm ─► (N, K, M) 矩阵乘 ─► 选一个 gemm_kacc 实例（共 10 个）
                         │  K ≤ 128 的小层补零后放到更大 K 的实例上（merge_small_k）
                         │  kernel_source：src/aiednn/gemm + src/aiednn/conv2d 模板 ─► 一个 aiek 源文件
                         │  emod.offload_group(...)：20 个 aiednn_conv + 1 个 aiednn_dense 调用点
                                         ▼
                emod.build(target="a78", runtime_io=[("input", ...)]) ─► main.elf（123 MB，输入在 .aiehlc_io 段）

板上：aiehlc_elfio.py patch 换输入 ─► apppaltest.py --netload ─► 输出转储（1000 个 fp32 logits）─► 和 onnxruntime 比
```

- **conv 怎么上 AIE：** A78 上做 im2col 把 conv 变成矩阵乘（[`src/aiednn/conv2d/kacc_gemm_pad_host.cc`](../../src/aiednn/conv2d/kacc_gemm_pad_host.cc)，补零到实例尺寸），AIE 上跑 `gemm_kacc`（[`src/aiednn/gemm/gemm_kacc_kernel.cc`](../../src/aiednn/gemm/gemm_kacc_kernel.cc)）：bf16 乘法、fp32 累加，K 方向在 core 内累加。
- **全连接层：** 最后的 Gemm（512 → 1000）同样走 gemm_kacc（`aiednn_dense` 调用点）。
- **一次推理 147 次 AIE 启动**，warm launch（kernel 只加载一次）+ control packet AOT + KCO（[`src/aiednn/gemm/ctrlpkt_aot_pragmas.h`](../../src/aiednn/gemm/ctrlpkt_aot_pragmas.h)、[`ctrlpkt_kco_pragma.h`](../../src/aiednn/gemm/ctrlpkt_kco_pragma.h)）。

用到的 aiegraph 模块：[`aiegraph/cnn.py`](../../src/aiegraph/cnn.py)（CNN 的组 offload）、[`aiegraph/boardmain.py`](../../src/aiegraph/boardmain.py)（板上 main、输入/参考头文件、驱动胶水）。aiegraph 的通用用法见 [aiegraph 教程](../../doc/tutorial_aiegraph.md)。

---

## 1. 准备

和 [UNet 教程](tiny-sdm-unet.md) 第 1–3 节一样：检出 `unet_m4` 分支、链接 BSP workspace、编 aiehlc。下面用这些变量：

```bash
export AIEHLC=/scratch/staff/huaj/aiehlc/aiehlcopensource/aiehlchj
export T=$AIEHLC/.worktrees/unet_m4                  # unet_m4 分支的 worktree
export PY=/scratch/staff/huaj/aiehlc/.venv/bin/python3
export WORK=$HOME/resnet18_work; mkdir -p $WORK
export ONNX=/scratch/staff/huaj/modelzoo/artifacts/resnet18/onnx/resnet18_fp32_op17.onnx
unset AR                                             # make 会把 $AR 当成归档工具
```

模型是 torchvision 的 ResNet18 导出的 fp32 ONNX（opset 17）。

---

## 2. 准备一张真实图片

预处理和 torchvision 的 ImageNet 评估一致：短边缩放到 256（双线性）、中心裁 224、/255、减均值除方差、NCHW fp32。这里用 imagenette 验证集里的一张丁鲷（tench，类别 0）：

```bash
IMG=/scratch/staff/huaj/modelzoo/data/imagenette_dl/imagenette2-320/val/n01440764/n01440764_10040.JPEG
$PY - "$IMG" $WORK/img0.bin $ONNX <<'PY'
import sys, numpy as np, onnxruntime as ort
from PIL import Image
img, out, onnx = sys.argv[1:]
im = Image.open(img).convert("RGB"); w, h = im.size; s = 256 / min(w, h)
im = im.resize((max(256, round(w * s)), max(256, round(h * s))), Image.BILINEAR); w, h = im.size
l, t = (w - 224) // 2, (h - 224) // 2
mean, std = np.array([0.485, 0.456, 0.406], np.float32), np.array([0.229, 0.224, 0.225], np.float32)
a = (np.asarray(im.crop((l, t, l + 224, t + 224)), np.float32) / 255.0 - mean) / std
x = np.ascontiguousarray(a.transpose(2, 0, 1)[None], np.float32); x.tofile(out)
sess = ort.InferenceSession(onnx, providers=["CPUExecutionProvider"])
ref = sess.run(None, {sess.get_inputs()[0].name: x})[0].reshape(-1)
ref.astype("<f4").tofile(out.replace(".bin", "_ort.bin"))
print("ORT top-5", np.argsort(-ref)[:5].tolist())
PY
```

得到 `$WORK/img0.bin`（602112 B，板子的输入）和 `$WORK/img0_ort.bin`（onnxruntime fp32 的 1000 个 logits，板上结果的参考）。

---

## 3. 用户侧的全部源码：build_resnet18.py

这就是 [`example/aiegraph/resnet18/build_resnet18.py`](../../example/aiegraph/resnet18/build_resnet18.py) 的全文。它只声明模型的输入、参考输出和选项；offload 是一行 `cnn.offload_all(...)`。

```python
#!/usr/bin/env python3
"""ResNet18 (ONNX, 1x3x224x224 -> 1000 logits) -> aiegraph -> A78 main.elf, every Conv and the Gemm on the AIE.

  $PY build_resnet18.py --onnx resnet18_fp32_op17.onnx --out <dir> [--input in.bin] [--ctrlpkt aot-kco]

This file only declares the model's input, its reference and the options. aiegraph does the rest:
  aiegraph.cnn        every Conv / weighted Gemm -> one aiegraph group on a bf16 gemm_kacc instance
                      (templates in src/aiednn/{gemm,conv2d}), archive on a carrier layer;
  aiegraph.boardmain  board main.c, inputs header, reference header, driver glue;
  emod.build          main.elf with the input in the .aiehlc_io section (patch it with aiehlc_elfio.py).
The build's own input is --input (raw fp32 NCHW) or a seeded N(0,1) tensor; its onnxruntime output is the
reference the board compares against when it runs that input. The build log prints
'OFFLOAD resnet18 conv=n/N gemm=m/M AIE_MAC_SHARE=x%'.
"""
import argparse
import os
import shutil
import sys
import zlib
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parents[2] / "src")]   # this checkout's aiegraph / aiednn


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--onnx", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--input", type=Path, default=None, help="raw fp32 1x3x224x224 (default: seeded N(0,1))")
    ap.add_argument("--ctrlpkt", choices=("off", "aot", "aot-kco"), default="aot-kco", help="control-packet plan")
    ap.add_argument("--launch-opt", choices=("cold", "warm"), default="warm", help="warm: kernels loaded once")
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args(argv)
    log = lambda msg: print(f"[build_resnet18] {msg}", flush=True)
    import onnx
    import onnxruntime as ort
    import aiegraph as ag
    from aiegraph import boardmain as bm, cnn
    bm.profiling_env()
    opts = bm.launch_env(a.launch_opt)
    log(f"launch_opt={a.launch_opt} ({opts}) ctrlpkt={a.ctrlpkt} defs={os.environ.get('EXTRA_DEFS', '')}")
    out = a.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    root = Path(ag.__file__).resolve().parents[2]
    os.environ.setdefault("AIEHLC_ROOT", str(root))
    os.environ["AIEGRAPH_PARAMS_INPLACE"] = "1"
    os.environ["AIEGRAPH_WDTYPE"] = "bf16"
    os.environ.setdefault("HOSTCOMPILE_HOST_OPT", "-O2")

    # the model's input and its x86 reference (onnxruntime, fp32)
    shape = (1, 3, 224, 224)
    x = (np.fromfile(a.input, dtype="<f4") if a.input else
         np.random.default_rng(0).standard_normal(shape).astype("<f4")).reshape(shape)
    sess = ort.InferenceSession(str(a.onnx), providers=["CPUExecutionProvider"])
    ref = sess.run(None, {sess.get_inputs()[0].name: x})[0].astype("<f4").reshape(-1)
    inb = out / "input.bin"
    x.tofile(inb)
    hdr = out / "resnet18_inputs.h"
    crc = bm.inputs_header(hdr, "resnet18", [("data", "input", x.reshape(-1))],
                           dict(AIE_MODE="conv,gemm", WDTYPE="bf16"), origin=inb)
    assert crc == "%08x" % zlib.crc32(x.tobytes())
    src = f"input {a.input.name}" if a.input else "seeded N(0,1) input"
    bm.ref_header(out / "p1_ref.h", "resnet18", ref, crc, f"ort/{src}")
    log(f"input {src} crc32={crc}; ORT reference top-1={int(ref.argmax())} ({ref.size} logits)")

    m = ag.load(str(a.onnx.resolve()))
    log(f"loaded {m.name}: {m.input_shapes} -> {m.output_shape}, {sum(m.ops.values())} nodes")
    shutil.rmtree(out / "ag_work", ignore_errors=True)
    emod = m.convert(target="a78", quantize="fp32", image=str(inb), workdir=out / "ag_work", verbose=a.verbose)
    log(f"converted -> {emod.workdir}")
    bm.cross_toolchain(root, log=log)
    res = cnn.offload_all(emod, onnx.load(str(m.onnx_path)), out, warm=a.launch_opt == "warm",
                          ctrlpkt=a.ctrlpkt, verbose=a.verbose, log=log)
    log(f"OFFLOAD resnet18 conv={res['conv'][0]}/{res['conv'][1]} gemm={res['gemm'][0]}/{res['gemm'][1]} "
        f"AIE_MAC_SHARE={res['mac_share']:.1f}%")
    log(f"aie_launches_planned={res['launches']} instances={len(res['shapes'])} kernel={res['kernel']}")
    bm.append_defines(hdr, "resnet18", res["defs"])

    main_c = bm.main_c("resnet18", ["input"], aie=True, runtime_io=True)
    rep = emod.build(target="a78", verbose=a.verbose, main_c=main_c,
                     extra_sources=[bm.cmp_source(), out / "p1_ref.h", hdr],
                     extra_cflags=f"{bm.CMP_CFLAGS} {bm.HEAP_LDFLAG}",
                     runtime_io=[("input", inb)], dump_out=True, launch=a.launch_opt)
    if not rep.ok:
        log(f"build FAILED: {rep.reason}")
        return 1
    for ln in rep.io_info.splitlines():
        log(ln)
    got = cnn.check_driver((Path(rep.elf).parent / "graph_driver.c").read_text(), res)
    log(f"linked driver: {got[0]} aiednn_conv + {got[1]} aiednn_dense call sites")
    elf, wb = out / "main.elf", Path(rep.elf).parent / "weights.bin"
    shutil.copy2(rep.elf, elf)
    if wb.is_file():
        shutil.copy2(wb, out / "weights.bin")
    log(f"built {rep.elf} -> {elf} ({elf.stat().st_size:,} B)")
    return 0


if __name__ == "__main__":
    try:
        rc = main()
    except SystemExit as e:
        rc = e.code if isinstance(e.code, int) else (print(e.code, flush=True) or 1)
    except Exception:
        import traceback
        traceback.print_exc()
        rc = 1
    print(f"GATE_RC={rc}", flush=True)
    sys.exit(rc)
```

几处要点：

| 代码 | 作用 |
|---|---|
| `bm.profiling_env()`、`bm.launch_env("warm")` | 打开板上 PROF 计时和 warm launch 的编译宏 |
| `bm.inputs_header(...)`、`bm.ref_header(...)` | 生成输入头文件（含 crc）和参考输出头文件，板上 `CMP` 用 |
| `ag.load(...).convert(target="a78", ...)` | ONNX → TVM C |
| `cnn.offload_all(emod, onnx_model, out, ...)` | 所有 Conv / Gemm 匹配成组、选实例、生成 kernel、声明 `offload_group`；返回覆盖统计 |
| `emod.build(..., runtime_io=[("input", inb)], dump_out=True)` | 链接 main.elf，输入放 `.aiehlc_io` 段，板上把输出转储出来 |
| `cnn.check_driver(...)` | 核对链接后的 `graph_driver.c` 里调用点数（20 conv + 1 dense） |

---

## 4. 构建

```bash
cd $T/example/aiegraph/resnet18
env -u AIEHLC_ROOT AIEHLC_BIN=$T/build/aiehlc $PY build_resnet18.py \
    --onnx $ONNX --out $WORK/build --input $WORK/img0.bin 2>&1 | tee $WORK/build.log
```

约 4 分钟（实测 236 s）。日志里应该能看到：

```
[build_resnet18] loaded resnet18_fp32_op17: {'data': [1, 3, 224, 224]} -> [1, 1000], 49 nodes
[build_resnet18] OFFLOAD resnet18 conv=20/20 gemm=1/1 AIE_MAC_SHARE=100.0%
[build_resnet18] aie_launches_planned=147 instances=10 kernel=.../build/aiek_src/aiek_cnn.cc
[build_resnet18] AIO input input off=128 bytes=602112 crc32=...
[build_resnet18] linked driver: 20 aiednn_conv + 1 aiednn_dense call sites
[build_resnet18] built .../main.elf -> $WORK/build/main.elf (122,801,136 B)
GATE_RC=0
```

`conv=20/20 gemm=1/1 AIE_MAC_SHARE=100.0%` 就是 AIE 覆盖比例：所有乘加都在 AIE 上。

---

## 5. 在板上跑

ELF 的输入在 `.aiehlc_io` 段里，换图片不用重新构建，用 `aiehlc_elfio.py` 改这个段即可：

```bash
cd $T
$PY src/tool/elfio/aiehlc_elfio.py patch $WORK/build/main.elf -o $WORK/run0.elf --in input=$WORK/img0.bin --dump-out
$PY script/test/apppaltest.py -board palmyra -host 10.23.224.213 -y --netload \
    --netload-wait 1800 --netload-reply-timeout 600 --netload-max-run-ms 300000 \
    $WORK/run0.elf > $WORK/board0.log 2>&1
grep -v DEBUG $WORK/board0.log | grep -E "graph_run|^LAUNCH warm"
$PY src/tool/elfio/aiehlc_elfio.py readout $WORK/board0.log -o $WORK/board0.npy --ref $WORK/img0_ort.bin
$PY -c "import numpy as np; b=np.load('$WORK/board0.npy').reshape(-1); r=np.fromfile('$WORK/img0_ort.bin','<f4'); \
print('board top-5', np.argsort(-b)[:5].tolist(), 'ort top-5', np.argsort(-r)[:5].tolist())"
```

- `patch` 打印 `PATCHED crc32=… outside_io_diff=0`：只改了输入段；
- `readout` 打印 `READOUT … crc_ok=1 cos=… cos_OK=1`：板上的 logits 和 onnxruntime 一致；
- 最后一行看 top-5 是否相同。

**注意：** `apppaltest.py` 就算板上打印了 FAIL 也会返回 0，要看日志里的结果行。

---

## 实测结果

2026-11-18，palmyra213，两张 imagenette 验证集图片（img0 = `n01440764_10040.JPEG`，img1 = `n02102040_1011.JPEG`）：

```
COMPARE resnet18 img0 board=palmyra213 in_crc32=8d91c275 aie_launches=147 graph_run=415.916ms cos=0.999956 top5_set=5/5 top1_same PASS
COMPARE resnet18 img1 board=palmyra213 in_crc32=709781c3 aie_launches=147 graph_run=414.546ms cos=0.999976 top5_set=5/5 top1_same PASS
```

| 指标 | 数值 |
|---|---|
| AIE 覆盖 | conv 20/20，gemm 1/1，按 MAC 算 100.0% |
| 每次推理的 AIE 启动 | 147（warm，`aot_fallbacks=0`） |
| `graph_run` | 415.9 ms / 414.5 ms |
| 对 onnxruntime fp32 的 cos | 0.999956 / 0.999976（bf16 乘法、fp32 累加） |
| top-5 | 两张图都 5/5 相同，top-1 相同 |

---

## 常见问题

| 现象 | 原因和办法 |
|---|---|
| kernel 链接报 DMb 放不下（`no 4096 B area`） | 某些"方形"实例（Ng == PX）的端口放不下；`aiegraph.cnn.instance()` 已自动换另一种启动宽度 |
| 某些 K 很小的层板上结果错（cos 只有 0.5–0.8） | gemm_kacc 在 K ≤ 128 时的已知问题；`merge_small_k` 已把这些层补零放到大 K 实例上。调试时加 `AIEDNN_SELFCHECK` 宏，每次启动都在 A78 上核对 |
| 结果行里夹着 `[DEBUG]` | `apppaltest` 的调试输出，`grep -v DEBUG` |

---

## 下一步：改哪里

| 想做什么 | 改哪里 |
|---|---|
| 改 conv 的 im2col / 补零 | [`src/aiednn/conv2d/kacc_gemm_pad_host.cc`](../../src/aiednn/conv2d/kacc_gemm_pad_host.cc) |
| 改 gemm_kacc kernel | [`src/aiednn/gemm/gemm_kacc_kernel.cc`](../../src/aiednn/gemm/gemm_kacc_kernel.cc) |
| 改实例的选法 | [`src/aiegraph/cnn.py`](../../src/aiegraph/cnn.py) 的 `instance()` / `merge_small_k()` |
| 换一个 CNN | 把 `build_resnet18.py` 的输入形状和名字改掉即可；带检测后处理的模型见 [YOLOv26](yolov26.md) |
