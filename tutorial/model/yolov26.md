# 用 aiegraph 把 YOLOv26（yolo26n）搬上 AIE：真实图片目标检测

本教程把 **yolo26n**（ultralytics YOLO26 nano，end2end 导出、COCO 训练好的权重，1×3×640×640）部署到 palmyra 板（A78 APU + AIE2PS）上：A78 跑 TVM 生成的 C 代码，**全部 102 个 Conv（含 8 个 depthwise）** 通过 aiegraph 的组 offload 跑在 AIE 上；在板上对 COCO 图片做检测，检测框和 onnxruntime 一致。

它和 [tiny-SD UNet](tiny-sdm-unet.md)、[ResNet18](resnet18.md) 用的是同一套 aiegraph。用户只写一个 `build_yolov26.py`（第 3 节贴出了全文），里面没有 C 文本：kernel 来自 [`src/aiednn/`](../../src/aiednn/) 的模板，驱动步骤、板上 main 和 `.aiehlc_io` 输入段都由 aiegraph 生成。

所有数字都是在 palmyra213（10.23.224.213）上实测的，见[实测结果](#实测结果)。

---

## 0. 它是怎么工作的

```
yolo26n_e2e_trained_op17.onnx
   │ aiegraph.cut.cut_before(model, "TopK")：在后处理尾巴（TopK / GatherElements）之前切开
   ▼
yolo26n_cut.onnx（输出 = pre-TopK 头 [1,84,8400]）─► ag.load ─► convert(target="a78") ─► TVM C
                                         │
     emod.match("conv", partition=..., groups=(1, "dw"))：每个 Conv（含 depthwise）对应到它的 TVM 调用
     aiegraph.cnn_chunk.plan_convs：输出像素按 MC=256 一块切开，每块一次 AIE 启动（共 1770 次）
     cnn_chunk.write_kernel：src/aiednn/gemm + src/aiednn/conv2d 模板 ─► 一个 aiek 源文件（放在一个 carrier 层上）
     cnn_chunk.offload_convs：102 个组，每组一个 cnn_aie_conv 步骤
                                         ▼
     emod.build(target="a78", runtime_io=[("images", ...)]) ─► main.elf（29 MB）

板上输出 pre-TopK 头 ─► host 上做 end2end 尾巴（取每个 anchor 最高分、阈值 0.25、取前 300）─► 检测框
```

- **为什么在 TopK 前切开：** TopK / GatherElements 是数据相关的选择，几乎不花算力，放在 host 上做；板上算的是整个骨干 + 检测头，输出 `[1,84,8400]`（前 4 行 xyxy，后 80 行是 sigmoid 后的类别分数）。
- **conv 怎么上 AIE：** 输出像素按 `--mc` 切块，每块在 A78 上 im2col，AIE 上跑 `gemm_kacc`（[`src/aiednn/gemm/gemm_kacc_kernel.cc`](../../src/aiednn/gemm/gemm_kacc_kernel.cc)），chunk 胶水在 [`src/aiednn/conv2d/kacc_chunk_conv.cc`](../../src/aiednn/conv2d/kacc_chunk_conv.cc)、[`kacc_chunk_board.cc`](../../src/aiednn/conv2d/kacc_chunk_board.cc)、[`kacc_chunk_entry.cc`](../../src/aiednn/conv2d/kacc_chunk_entry.cc)。
- **精度：** 默认 `--split 3`：把 fp32 拆成 bf16 hi/lo 做三次 GEMM，结果接近 fp32（板上头的 cos = 1.000000）。`--split 1` 是纯 bf16，更快但误差大。

用到的 aiegraph 模块：[`aiegraph/cut.py`](../../src/aiegraph/cut.py)（按算子切图）、[`aiegraph/autopart.py`](../../src/aiegraph/autopart.py)（`inventory`：每个 Conv/MatMul 的 MAC 数）、[`aiegraph/cnn_chunk.py`](../../src/aiegraph/cnn_chunk.py)（分块 conv offload）、[`aiegraph/boardmain.py`](../../src/aiegraph/boardmain.py)。aiegraph 的通用用法见 [aiegraph 教程](../../doc/tutorial_aiegraph.md)。

---

## 1. 准备

和 [UNet 教程](tiny-sdm-unet.md) 第 1–3 节一样：检出 `unet_m4` 分支、链接 BSP workspace、编 aiehlc。下面用这些变量：

```bash
export AIEHLC=/scratch/staff/huaj/aiehlc/aiehlcopensource/aiehlchj
export T=$AIEHLC/.worktrees/unet_m4                  # unet_m4 分支的 worktree
export PY=/scratch/staff/huaj/aiehlc/.venv/bin/python3
export WORK=$HOME/yolo_work; mkdir -p $WORK
export ONNX=/scratch/staff/huaj/modelzoo/artifacts/yolo26/yolo26n_e2e_trained_op17.onnx
unset AR                                             # make 会把 $AR 当成归档工具
```

模型是 **训练好的** yolo26n（COCO 80 类），ultralytics end2end 导出、opset 17：`yolo26n_e2e_trained_op17.onnx`。输出 `output0` 是 `[1,300,6]`（x1, y1, x2, y2, score, class）。

---

## 2. 准备一张真实图片

预处理和 ultralytics 一致：保持比例缩放到 640，四周填灰（114）居中，RGB、/255、NCHW fp32。这里用 COCO val2017 的 `000000000139.jpg`（客厅）：

```bash
IMG=/scratch/staff/huaj/modelzoo/data/coco/images/val2017/000000000139.jpg
$PY - "$IMG" $WORK/img0.bin <<'PY'
import sys, numpy as np
from PIL import Image
img, out = sys.argv[1:]
im = Image.open(img).convert("RGB"); w, h = im.size; r = min(640 / w, 640 / h)
nw, nh = round(w * r), round(h * r); im = im.resize((nw, nh), Image.BILINEAR)
c = np.full((640, 640, 3), 114, np.uint8)
top, left = round((640 - nh) / 2 - 0.1), round((640 - nw) / 2 - 0.1)
c[top:top + nh, left:left + nw] = np.asarray(im)
np.ascontiguousarray((c.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]).tofile(out)
PY
```

得到 `$WORK/img0.bin`（4915200 B）。

---

## 3. 用户侧的全部源码：build_yolov26.py

这就是 [`example/aiegraph/yolov26/build_yolov26.py`](../../example/aiegraph/yolov26/build_yolov26.py) 的全文。它只声明：后处理尾巴从哪里开始（`TAIL_OP = "TopK"`）、模型输入、哪些调用上 AIE（每个匹配到的 conv 组）。

```python
#!/usr/bin/env python3
"""yolo26n (end2end ONNX, trained) -> aiegraph -> A78 main.elf, every Conv (incl. depthwise) on the AIE.

  $PY build_yolov26.py --onnx yolo26n_e2e_trained_op17.onnx --out <dir> [--input images.bin]

This file only declares the model: where its post-processing tail starts (TopK: cut before it, the
TopK / GatherElements selection runs on the host from the pre-TopK head), its input, and which calls go
to the AIE (every matched conv group, aiegraph.cnn_chunk). aiegraph generates the kernel source (src/aiednn),
the driver steps, the board main and the .aiehlc_io runtime-input section.
"""
import argparse
import json
import os
import shutil
import sys
import zlib
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parents[2] / "src")]   # this checkout's aiegraph (gate: plain $PY)
MODEL, KPREFIX, ENTRY, TAIL_OP = "yolo26n", "cnn", "aiek_cnnconv", "TopK"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--onnx", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--input", type=Path, help="fp32 [1,3,640,640] build input (default: a fixed synthetic image)")
    ap.add_argument("--mc", type=int, default=256, help="output pixels per AIE launch")
    ap.add_argument("--split", type=int, default=3, choices=(1, 3), help="3: bf16 hi/lo GEMM (~fp32), 1: bf16")
    ap.add_argument("--selfcheck", action="store_true", help="debug: every AIE launch checked on the A78")
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args(argv)
    log = lambda msg: print(f"[build_yolov26] {msg}", flush=True)
    import onnx
    import onnxruntime as ort
    import aiegraph as ag
    from aiegraph import boardmain as bm, cnn_chunk as cnn
    from aiegraph.autopart import inventory
    from aiegraph.cut import cut_before

    out = a.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    root = Path(ag.__file__).resolve().parents[2]
    os.environ.setdefault("AIEHLC_ROOT", str(root))
    wdt = os.environ["AIEGRAPH_WDTYPE"] = "fp32" if a.split == 3 else "bf16"   # AIE: bf16 MACs, fp32 acc
    os.environ["AIEGRAPH_PARAMS_INPLACE"] = "1"            # weights read in place from the linked blob
    os.environ.setdefault("HOSTCOMPILE_HOST_OPT", "-O2")
    bm.profiling_env()
    opts = bm.launch_env("warm")
    log(f"aiegraph from {ag.__file__}; onnx {a.onnx}; launch warm ({opts})")

    # 1. the offloaded model: everything before the TopK tail (output = the pre-TopK head)
    cut, tensor = cut_before(onnx.load(str(a.onnx)), TAIL_OP, out=out / f"{MODEL}_cut.onnx")
    log(f"cut before {TAIL_OP}: output {tensor} {[d.dim_value for d in cut.graph.output[0].type.tensor_type.shape.dim]}"
        f", {len(cut.graph.node)} nodes")
    iname = cut.graph.input[0].name
    shape = [d.dim_value for d in cut.graph.input[0].type.tensor_type.shape.dim]
    if a.input:
        x = np.fromfile(a.input, "<f4").reshape(shape)
    else:   # letterbox gray with a smooth pattern, in [0, 1]
        yy, xx = np.meshgrid(np.linspace(0, 1, shape[2]), np.linspace(0, 1, shape[3]), indexing="ij")
        x = np.stack([0.45 + 0.4 * np.sin(6.0 * xx + c) * yy for c in range(shape[1])])[None].astype("<f4")
    xin = out / "images.bin"
    x.astype("<f4").tofile(xin)
    crc = "%08x" % zlib.crc32(x.astype("<f4").tobytes())
    ref = ort.InferenceSession(str(out / f"{MODEL}_cut.onnx"), providers=["CPUExecutionProvider"]).run(None, {iname: x})[0]
    log(f"input {iname} {shape} crc32={crc}; ORT fp32 head {list(ref.shape)} (board CMP reference)")

    # 2. convert (TVM, per-layer C) and match every Conv, depthwise included, to its TVM call
    shutil.rmtree(out / "ag_work", ignore_errors=True)
    emod = ag.load(str(out / f"{MODEL}_cut.onnx")).convert(target="a78", quantize="fp32", image=str(xin),
                                                           workdir=out / "ag_work", verbose=a.verbose)
    bm.cross_toolchain(root, log=log)
    inv = inventory(cut)
    rows = [dict(idx=i, node=r["node"], weight=r["weight"], cls="conv" if r["op"] == "Conv" else "linear",
                 mkn=[[r["m"], r["k"], r["n"]]]) for i, r in enumerate(inv) if r["cls"] != "attn"]
    groups = emod.match("conv", partition={"layers": rows}, model=cut, groups=(1, "dw"))
    nconv = sum(1 for n in cut.graph.node if n.op_type == "Conv")
    ndw = sum(1 for r in inv if r["op"] == "Conv" and r["why"])
    plans = cnn.plan_convs(groups, mc=a.mc, split=a.split)
    total = sum(r["macs"] for r in inv)
    aie = sum(r["macs"] for r in inv if r["node"] in {p.onnx for p in plans})
    for p in plans:
        log(f"  aie conv {p.cid:3d} {p.onnx} -> call {p.call} M={p.M} N={p.N} K={p.K} s={p.sh} p={p.ph} dw={p.dw} "
            f"act={p.act} -> {p.Np}x{p.Kk}x{p.mc} split={p.split} launches={p.launches}")
    log(f"matmul (host): {sum(1 for r in inv if r['op'] != 'Conv')} nodes, "
        f"{sum(r['macs'] for r in inv if r['op'] != 'Conv')} MACs")
    print(f"OFFLOAD {MODEL} conv={len(plans)}/{nconv} dw={sum(p.dw for p in plans)}/{ndw} "
          f"AIE_MAC_SHARE={100.0 * aie / total:.1f}%", flush=True)

    # 3. kernel source on a carrier layer, one group step per conv, the PROF driver glue, AIE archive
    lidx, car = cnn.pick_carrier(plans, Path(emod.workdir) / "layers")
    proto = emod.offload_prototype(lidx, entry=ENTRY)
    kcc = cnn.write_kernel(out / "aiek_src", proto.spec(), plans, kprefix=KPREFIX, entry=ENTRY, warm=True,
                           selfcheck=a.selfcheck)
    log(f"carrier layer {lidx} ({car.symbol}); kernel source {kcc}")
    planned = cnn.offload_convs(emod, groups, plans, kcc, kprefix=KPREFIX, log=log)
    emod.driver_template(**bm.driver_glue(kprefix=KPREFIX, bf16_expand=wdt == "bf16"))
    rep = emod.offload(f"layer{lidx}", str(kcc), check="group", verbose=a.verbose)
    log(f"offload: {rep}")
    if not any(L.get("target") == "aie" and L.get("archive") for L in
               json.loads((Path(emod.workdir) / "layers" / "partition.json").read_text())["layers"]):
        log("offload FAILED: the carrier layer has no AIE archive")
        return 1

    # 4. board main + inputs header + reference, link
    hdr = out / f"{MODEL}_inputs.h"
    gin = emod.input_names      # the converted graph's name of the ONNX input
    bm.inputs_header(hdr, MODEL, [(gin[0], iname, x.reshape(-1))], dict(AIE_MODE="conv", WDTYPE=wdt), origin=xin)
    bm.append_defines(hdr, MODEL, dict(LAUNCH_WARM=1, LAUNCH_OPTS=bm.LAUNCH_OPTS_WARM, CTRLPKT="off",
                                       AIE_PLANNED_LAUNCHES=planned, AIE_PLANNED_CALLS=len(plans),
                                       TOTAL_GEMM_MACS=(total, "ULL")))
    refh = bm.ref_header(out / "p1_ref.h", MODEL, ref, crc, data="build-input")
    main_c = bm.main_c(MODEL, [iname], aie=True, runtime_io=True, kprefix=KPREFIX)
    rep = emod.build(target="a78", verbose=a.verbose, main_c=main_c, extra_sources=[bm.cmp_source(), refh, hdr],
                     extra_cflags=f"{bm.CMP_CFLAGS} {bm.HEAP_LDFLAG}", runtime_io=[(iname, xin)], dump_out=True,
                     launch="warm")
    if not rep.ok:
        log(f"build FAILED: {rep.reason}")
        return 1
    for ln in rep.io_info.splitlines():
        log(ln)
    drv = (Path(rep.elf).parent / "graph_driver.c").read_text()
    sites = drv.count(f"    {KPREFIX}_aie_conv(")
    if sites != len(plans):
        log(f"build FAILED: linked driver has {sites} {KPREFIX}_aie_conv call sites, expected {len(plans)}")
        return 1
    log(f"linked driver: {sites} call sites -> {KPREFIX}_aie_conv (AIE), {planned} launches planned")
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

分四步：

| 步骤 | aiegraph 调用 | 作用 |
|---|---|---|
| 1. 切图 | `cut_before(model, "TopK")` | 得到 `yolo26n_cut.onnx`，输出 pre-TopK 头；用 onnxruntime 在它上面算板上 CMP 的参考 |
| 2. 转换、匹配 | `ag.load(...).convert(...)`、`emod.match("conv", ..., groups=(1, "dw"))` | 每个 Conv（含 depthwise 组卷积）对应到 TVM 调用 |
| 3. kernel、组 | `cnn.plan_convs`、`cnn.write_kernel`、`cnn.offload_convs`、`emod.offload(..., check="group")` | 切块计划、kernel 源、102 个组、组级 x86 校验 |
| 4. 链接 | `bm.inputs_header`、`bm.ref_header`、`bm.main_c`、`emod.build(..., runtime_io=...)` | 板上 main、输入/参考头文件、main.elf；最后核对 `graph_driver.c` 里 `cnn_aie_conv(` 调用点数 |

---

## 4. 构建

```bash
cd $T/example/aiegraph/yolov26
env -u AIEHLC_ROOT AIEHLC_BIN=$T/build/aiehlc $PY build_yolov26.py \
    --onnx $ONNX --out $WORK/build --input $WORK/img0.bin 2>&1 | tee $WORK/build.log
```

日志里应该能看到：

```
[build_yolov26] cut before TopK: output /model.23/Concat_3_output_0 [1, 84, 8400], 365 nodes
OFFLOAD yolo26n conv=102/102 dw=8/8 AIE_MAC_SHARE=97.8%
[build_yolov26] carrier layer 36 (tvmgen_default_fused_nn_contrib_conv2d_NCHWc_add); kernel source .../aiek_src/aiek_cnnconv.cc
[build_yolov26] offload: offload  READY
[build_yolov26] AIO input images off=128 bytes=4915200 crc32=...
[build_yolov26] linked driver: 102 call sites -> cnn_aie_conv (AIE), 1770 launches planned
[build_yolov26] built .../main.elf -> $WORK/build/main.elf (29,393,352 B)
GATE_RC=0
```

`conv=102/102 dw=8/8 AIE_MAC_SHARE=97.8%` 是 AIE 覆盖比例：所有 conv 都在 AIE 上；剩下 2.2% 的 MAC 是检测头里 attention 的 MatMul，还在 A78 上。

---

## 5. 在板上跑并解码检测框

```bash
cd $T
$PY src/tool/elfio/aiehlc_elfio.py patch $WORK/build/main.elf -o $WORK/run0.elf --in images=$WORK/img0.bin --dump-out
$PY script/test/apppaltest.py -board palmyra -host 10.23.224.213 -y --netload \
    --netload-wait 1800 --netload-reply-timeout 600 --netload-max-run-ms 900000 \
    $WORK/run0.elf > $WORK/board0.log 2>&1
grep -v DEBUG $WORK/board0.log | grep -E "graph_run|^LAUNCH warm"
```

再在 host 上算 onnxruntime 的参考头，读回板上的头，两边都按 end2end 尾巴解码成框后比对：

```bash
$PY - $WORK/build/yolo26n_cut.onnx $WORK/img0.bin $WORK/board0.log <<'PY'
import subprocess, sys, numpy as np, onnxruntime as ort
onnx, inb, log = sys.argv[1:]
x = np.fromfile(inb, "<f4").reshape(1, 3, 640, 640)
s = ort.InferenceSession(onnx, providers=["CPUExecutionProvider"])
ref = s.run(None, {s.get_inputs()[0].name: x})[0].reshape(84, 8400)
ref.astype("<f4").tofile(inb.replace(".bin", "_ort.bin"))
subprocess.run([sys.executable, "src/tool/elfio/aiehlc_elfio.py", "readout", log, "-o", log + ".npy",
                "--ref", inb.replace(".bin", "_ort.bin")], check=True)
brd = np.load(log + ".npy").reshape(84, 8400)

def boxes(h, conf=0.25, maxdet=300):            # end2end tail: best class per anchor, top-300, score >= conf
    sc, cl = h[4:].max(0), h[4:].argmax(0)
    k = np.argsort(-sc)[:maxdet]; k = k[sc[k] >= conf]
    return h[:4, k].T, cl[k], sc[k]

def iou(a, b):
    w = np.clip(np.minimum(a[2], b[2]) - np.maximum(a[0], b[0]), 0, None)
    h = np.clip(np.minimum(a[3], b[3]) - np.maximum(a[1], b[1]), 0, None)
    i = w * h; return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i)

rb, rc, _ = boxes(ref); bb, bc, _ = boxes(brd)
cos = float(ref.ravel() @ brd.ravel() / np.linalg.norm(ref) / np.linalg.norm(brd))
ok = sum(any(bc[j] == rc[i] and iou(rb[i], bb[j]) >= 0.9 for j in range(len(bb))) for i in range(len(rb)))
print(f"head_cos={cos:.6f} boxes matched {ok}/{len(rb)} (board {len(bb)} boxes)")
PY
```

- `readout … cos_OK=1`：板上的头和 onnxruntime 一致；
- `boxes matched N/N`：每个参考框都有一个类别相同、IoU ≥ 0.9 的板上框。

**注意：** 一次推理约 9.5 s，`--netload-max-run-ms` 要给够（上面是 900000）；`apppaltest.py` 就算板上打印了 FAIL 也会返回 0。

---

## 实测结果

2026-11-18，palmyra213，两张 COCO val2017 图片（img0 = `000000000139.jpg`，img1 = `000000000785.jpg`），`--split 3 --mc 256`：

```
COMPARE yolo26n img0 board=palmyra213 in_crc32=34eab628 aie_launches=1770 graph_run=9495.027ms head_cos=1.000000 boxes matched 12/12 PASS
COMPARE yolo26n img1 board=palmyra213 in_crc32=b8aa040c aie_launches=1770 graph_run=9500.179ms head_cos=1.000000 boxes matched 2/2 PASS
```

| 指标 | 数值 |
|---|---|
| AIE 覆盖 | conv 102/102（depthwise 8/8），按 MAC 算 97.8%（其余是 attention MatMul，在 A78 上） |
| 每次推理的 AIE 启动 | 1770（warm） |
| `graph_run` | 9495.0 ms / 9500.2 ms |
| 对 onnxruntime fp32 的头 cos | 1.000000 / 1.000000 |
| 检测框 | 12/12、2/2 全部匹配（类别相同、IoU ≥ 0.9），没有多出的框 |

时间主要花在 1770 次小启动（每块 256 个输出像素）和 A78 上的 im2col；加大 `--mc` 能减少启动次数。

---

## 下一步：改哪里

| 想做什么 | 改哪里 |
|---|---|
| 改 chunk 的切法 / 每次启动的像素数 | `--mc`，以及 [`src/aiegraph/cnn_chunk.py`](../../src/aiegraph/cnn_chunk.py) 的 `plan_convs` |
| 改 chunk 的主机胶水 | [`src/aiednn/conv2d/kacc_chunk_conv.cc`](../../src/aiednn/conv2d/kacc_chunk_conv.cc) |
| 改 gemm_kacc kernel | [`src/aiednn/gemm/gemm_kacc_kernel.cc`](../../src/aiednn/gemm/gemm_kacc_kernel.cc) |
| 调试某次启动算错 | `--selfcheck`：每次 AIE 启动都在 A78 上核对 |
| 换一个带后处理尾巴的模型 | 改 `TAIL_OP`（切开的算子）和输入；分类模型见 [ResNet18](resnet18.md) |
