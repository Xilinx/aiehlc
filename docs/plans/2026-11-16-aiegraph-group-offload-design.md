# 设计：用 aiegraph 的"成组 offload"部署 tiny-SD UNet

日期：2026-11-16
状态：草案，待评审
相关：`tutorial/model/tiny-sdm-unet.md`（现在的做法，能出橘子）、`docs/plans/2026-09-09-aiegraph-primitive-fusion-offload-design.md`（ResNet 上的融合 offload，不同的问题）

---

## 1. 目标

用户拿到一个网络，只调用 aiegraph 的函数，就能编出一个能在 VEK385（palmyra）上生成橘子的 ELF：

```
ONNX → ag.load → convert("a78") → [Python 模板生成 kernel] → emod.offload_group(...) → emod.build → main.elf
```

- **kernel 模板**继续用现在的 Python 生成（`unet_aie.py` 的 gemm_kacc、`fa_attn.py` 的 flash-attention）。
- **offload**（哪个调用换成什么、换得对不对）由 aiegraph 做，不再由例子目录里的脚本改写 `graph_driver.c`。
- **性能**不能变差：生成出来的驱动要和现在的等价，板上 graph_run ≈ 1.295 s、258 次 launch、cos 1.000000。

---

## 2. 现状：aiegraph 只用了一半

`example/aiegraph/unet_bk_sdm_tiny/build_unet.py` 的流程：

| 步骤 | 谁做的 |
|---|---|
| 载入 ONNX、转成 TVM C（`ag.load`、`convert("a78")`） | aiegraph |
| 编出 AIE 库 | aiegraph 的 `emod.offload`，但**只用在 1 个"载体层"上**，目的只是让 aiegraph 调 aiehlc 把整个 `.cc` 编成库 |
| 另外 155 个调用点换成 AIE | `build_unet.py` 把 `arm_build.generate_driver` 换掉（猴子补丁），在生成好的 `graph_driver.c` 上用正则改写：`unet_aie.patch_driver()`、`fa_attn`、`keep_layout_rt.keep_layout_driver()` |
| bf16 权重原地读、按 kacc 块布局预排 | 正则删掉 `graph_bf16_expand`，`prelayout_weights()` 改写 weights.bin |
| x86 校验 | calibrate、emulate 被替换成空函数，`offload(force=True)` |
| 链接出 ELF（`emod.build`） | aiegraph |

所以用户现在并没有真的"调用 aiegraph 来 offload"。

---

## 3. 概念澄清：我们操作的是什么"区域"

讨论中"区域整合""区域拆分"这些说法容易混淆，这里统一说清楚。

- **TVM 融合函数**：TVM 把几个 ONNX 算子融成一个函数，比如 `reshape_transpose_multiply_reshape`，对应 `graph_driver.c` 里的**一次调用**。这是 TVM 已经定好的东西，**我们既不拆它，也不新造一个更大的函数。**
- **替换**：永远是**一个调用对一个调用**，要么换成一个 AIE 侧的函数，要么删掉。
- **组（group）**：几个**完整的** TVM 调用，它们的替换函数合起来才等于原来那几个调用合起来的结果。组只是一个**验收单位**："这几个替换要一起检查"。

### 3.1 为什么 aiegraph 现在的一换一不够用

aiegraph 现在的 offload（`aiekernel/flow.py`、`arm_build.py:875`）有三条隐含要求：

1. 按**函数名**替换：同名函数的所有调用一起换；
2. 替换函数拿到的就是原调用的参数；
3. 每个替换函数单独和原 TVM 函数比对，结果要一致（calibrate）。

以 UNet 第 0 组 attention 为例（`graph_driver.c` 里的调用 [31]–[37]，都来自 TVM）：

| 调用 | 原 TVM 函数 | 原来算什么 | 现在替换成 | 单独比对 |
|---|---|---|---|---|
| [31] | `reshape_transpose_multiply_reshape` | 写出转置并缩放后的 Q | 从 linear 的输出 `tensors[58]`（`[1][1024][320]`）按 stride 把 Q 打包进 kernel 的暂存区，缩放 0.398 在打包时乘进去 | ✗ 不写出 Q |
| [32] | `…multiply_reshape_transpose` | 写出 Kᵀ | 打包减去均值后的 K | ✗ |
| [33] | `nn_batch_matmul` | 写出 scores | **删掉** | ✗ |
| [34] | `nn_softmax` | 写出 P | **删掉** | ✗ |
| [35] | `reshape_transpose_reshape_transpose` | 写出 Vᵀ | 打包 V | ✗ |
| [36] | `nn_batch_matmul_1` | 写出多头布局的 O | 启动 flash-attention，把 O 按 stride **直接写进 `tensors[70]`**（`[1024][320]`，[37] 的输出、下一个 linear 的输入） | ✗ |
| [37] | `reshape_transpose_reshape` | 写出合并后的 O | **删掉** | ✗ |

每个调用都是一换一，但：

- 每个替换**单独看都不对**：中间结果（Q、Kᵀ、scores、P、Vᵀ）根本没算出来，只有整组算完后 `tensors[70]` 才和 TVM 一致；
- 有的替换是"删掉"；
- [36] 写的是 [37] 的输出 buffer，用的不是它自己的参数；
- 同一个 TVM 函数名会被多个组共用（形状相同就共用），所以必须按**调用**换，不能按名字换。

这就是为什么需要"组"：替换方式还是一个一个来，但**验收要按组做**。

### 3.2 收益为什么值得

这些都是板上实测的数（当时还没开 KCO）：

| 改动 | graph_run |
|---|---|
| attention 上 AIE，但不融合（438 次 launch） | 2,680 ms |
| flash-attention 融合（softmax 删掉，258 次 launch） | 1,693 ms |
| 再加 keep-layout（删掉 56 个 layout 调用） | 1,650 ms |
| 现在（再加 KCO） | 1,295 ms |

如果 aiegraph 只能做朴素的一换一，就做不了融合，每次 UNet 调用大约慢 1 秒。

---

## 4. 现在的驱动直改是怎么做到的

`unet_aie.py`、`fa_attn.py`、`keep_layout_rt.py` 直接改 `graph_driver.c` 的文本，思路是：

- **不挪动任何调用**：每一步都留在原来某个调用的位置，所以 TVM 的存储计划仍然成立；
- **中间状态放在 kernel 自己的静态暂存区**（`g_fa_qk`、`g_fa_v`），不经过 TVM 的 buffer；
- 用 assert 保证这样改是对的：
  1. 被删掉的函数确实只是排列：用正则看 TVM 函数的 C 代码（`_tvm_body`），stride 必须是 H·D 这类期望值；
  2. 改从源头读是安全的：被跳过的那个中间张量只有这一个消费者（`_consumers(...) == [iq]`）；
  3. 提前写是安全的：[36] 写 [37] 的输出，中间没人碰这个 buffer（`_untouched`）；
  4. 顺序对：Q、K 的打包在 QKᵀ 之前，V 的打包在 PV 之前，每组恰好删掉一个 softmax；
  5. 有一条不满足，构建就失败。

同样的思路也用在别处：
- **conv**：AIE conv 直接写出下一层要的布局（`g_kl_y_14` 换了一个 shape 标签），后面的 layout 调用删掉；
- **残差加法**：并进 linear 的 scatter；
- **temb**：SiLU 的缩放在编译时折进权重，预排好写进 weights.bin。

**这些做法的弱点：**
- 匹配的是生成出来的 C 文本，`arm_build.py` 的输出格式一改，正则就可能失效（cherry-pick GIN 时遇到过冲突）；
- TVM 函数的语义靠正则看代码判断，而且函数名的模式（`Q_FN`、`K_FN`）是 UNet 专用的；
- 各个步骤之间的约定（role 2/3/5/6 的顺序、静态缓冲）只写在 C 代码里，aiegraph 不知道；
- 没有 x86 校验，只能靠上板跑整网的 cos 来兜底。

下面的设计**保留"不挪动调用 + kernel 自管中间状态"这个已经在板上验证过的模型**，只是把匹配、放置和检查从正则改 C 文本，移到 aiegraph 里、在图上做。

---

## 5. 接口

### 5.1 原则

用户**不写"第 31 个调用换成什么"**，只声明三件事：

1. **哪一组**：用模式匹配找出来（attention、conv……），或者给出一组 ONNX 节点；
2. **组的边界数据**：组外流进来的张量、流出去的张量。由 aiegraph 算出来，并带上形状、布局和行距；
3. **kernel 分几步，每步读或写哪个边界张量。**

每一步放在哪个调用的位置、删掉哪些调用、检查和校验，都由 aiegraph 来做。

### 5.2 Python（用户写的）

```python
import aiegraph as ag
import unet_aie, fa_attn                      # Python 模板，留在例子目录

emod = ag.load("unet_sim.onnx").convert("a78")

# ---- attention：一组 = 7 个 TVM 调用 ----
groups = emod.match("attention")              # 18 组
g = groups[0]
g.calls        # [31, 32, 33, 34, 35, 36, 37]
g.q, g.k, g.v  # 组外流进来的张量：tensors[58/59/60]，[1][1024][320] fp32，ld=320，每个头占一个列块
g.o            # 流出去的张量：tensors[70]，[1024][320]
g.scale        # 0.398：TVM 融进转置函数里的缩放，由 matcher 从 Relay 图里取出来
g.cls          # 形状类别（多个组共用同一个 kernel 实例）

fa_src = fa_attn.generate(groups)             # 按 g.q.shape、ld 等生成 kernel 和各个步骤函数

for g in groups:
    emod.offload_group(g, fa_src, steps=[
        ag.step("fa_pack_q", reads=g.q, consts=dict(cls=g.cls, s=g.scale)),
        ag.step("fa_pack_k", reads=g.k, consts=dict(cls=g.cls, s=g.scale)),
        ag.step("fa_pack_v", reads=g.v, consts=dict(cls=g.cls)),
        ag.step("fa_run_o",  writes=g.o, consts=dict(cls=g.cls)),
    ])

# ---- conv / linear：一组只有 1 个调用、1 个步骤 ----
for g in emod.match("conv", partition="partition_p4a.json"):
    emod.offload_group(g, kacc_src, steps=[
        ag.step("unet_aie_conv", reads=g.x, writes=g.y,
                weights=g.w.as_bf16(layout="kacc"),     # 权重约定：bf16、预排，A78 不展开
                consts=g.geom)])                       # cid、stride、pad

# ---- keep-layout：AIE conv 调用 + 它后面的 layout 调用 = 一组 ----
for g in emod.match("conv+layout"):
    emod.offload_group(g, kacc_src, steps=[
        ag.step("unet_aie_conv", reads=g.x, writes=g.y_after_layout, weights=..., consts=...)])

# ---- temb：SiLU 缩放折进权重（编译期变换） ----
for g in emod.match("temb_dense"):
    emod.offload_group(g, kacc_src, steps=[...],
                       weight_transform=lambda w, s: (w * s).astype("bfloat16"))

emod.build("a78", runtime_io=..., layer_cflags=...)    # 和现在一样
```

所有情况都用同一套接口，区别只在一组有几个调用。

### 5.3 C（模板生成的 `.cc` 里）

所有步骤函数用同一种调用方式（ABI），并像现在的 `AIEK_SPEC` 一样声明读写什么：

```c++
AIEK_STEP(fa_pack_q, reads(1), writes(0), consts(cls, s))
int fa_pack_q(const aiek_buf *in, aiek_buf *out, const aiek_const *c);  /* 打包进 kernel 的静态暂存区 */

AIEK_STEP(fa_run_o, reads(0), writes(1), consts(cls))
int fa_run_o(const aiek_buf *in, aiek_buf *out, const aiek_const *c);   /* 启动 FA，按 stride 写出 O */
```

`aiek_buf` = 指针 + 形状 + 行距 + dtype，kernel 不需要猜布局。aiegraph 用 `AIEK_STEP` 核对 Python 侧的 `ag.step(...)` 声明是否一致。

---

## 6. aiegraph 内部做什么

| 环节 | 规则 |
|---|---|
| **匹配** | `match("attention")` 基于已有的 `partition.match_attention`（ONNX 层面），再对应到 TVM 调用：带权重的算子按权重指纹（`region.py` 已有），纯激活的算子按数据流。组必须由**完整的** TVM 调用组成，有调用一半在组里一半在组外就报错。融进去的常数（scale）从 Relay 图里取出来 |
| **边界** | 组内被读、但在组外产生的张量 = 输入；组内产生、组外有人读的张量 = 输出。都带上形状和布局 |
| **放置** | 读输入的步骤，放在组内最早读这个张量的调用的位置（此时数据一定还在）；写输出的步骤，放在它依赖的步骤之后。不挪动任何组外的调用，TVM 的存储计划保持不变 |
| **删除** | 组内其余调用全部删掉。前提：它们产生的中间张量组外没人读，否则报错 |
| **提前写** | 写输出的步骤如果比原来的产生者早（[36] 写 [37] 的输出），要求这段区间内没有调用碰这个 buffer，否则报错 |
| **纯排列判定** | 被删的调用如果是 layout/reshape/transpose，在 Relay 图上看它的算子，而不是用正则看 C 代码 |
| **权重** | `weights=...as_bf16(layout=...)`：这一层的权重在 weights.bin 里按 kernel 要的布局存成 bf16，驱动不再为它调 `graph_bf16_expand`。`weight_transform` 是编译期的变换钩子 |
| **x86 校验** | 在边界输入上按原顺序跑组内这串 TVM C 函数作为参考；再按放好的顺序在 kemu 里跑这些步骤；比较输出。只跑这一组，不跑整网，代价很小，所以不再需要 `force=True` |
| **构建** | 所有步骤都在一个 `.cc` 里，编成一个库，只有一个 `routing()`（GAP ea6615）。warm 启动、KCO、control packet 的 pragma 都不变 |
| **记录** | 每组写进 `layers/partition.json`：组 id、成员调用、每个成员的动作、步骤和常数。`offload_reset` 可以整组撤回 |

### 6.1 生成出来的驱动（和现在等价）

```c
/* [31] */ fa_pack_q(&(aiek_buf){tensors[58] ...}, 0, &(aiek_const){.cls = 0, .s = 0.398f});  /* group attn0 */
/* [32] */ fa_pack_k(...);
/* [33] [34] removed: group attn0 */
/* [35] */ fa_pack_v(...);
/* [36] */ fa_run_o(0, &(aiek_buf){tensors[70] ...}, ...);
/* [37] removed: group attn0 (o written by fa_run_o at [36]) */
```

和现在的 `unet_aie_attn_kl(4, 2, 0, tensors[58].data, 320, 0, 0, 0.398f)` 是一回事。

---

## 7. 对性能的影响

- **原则上没有影响。** 板上跑多快，只取决于最终的驱动和 kernel，跟是谁生成的无关。只要生成的驱动和现在等价（调用、权重原地读、一个库、warm 启动、KCO），ELF 就一样快。
- **只有缺功能、被迫退回时才会掉性能。** 不支持组 → 不能融合，约 +1 s；不支持 keep-layout → 约 +43 ms；权重退回 fp32 → 每次调用都要在运行时重排约 0.65 GB 的权重（没测过，但代价不会小）。
- x86 校验只让编译变慢，不影响运行时间。

---

## 8. 实施计划

**每一步的验收：**
1. 生成的 `graph_driver.c` 和基准等价（diff 只有命名和注释）；
2. 板上冒烟测试：graph_run ≈ 1.295 s、258 次 launch、cos 1.000000、`aot_fallbacks=0`；
3. 最后一步再跑一遍教程里的橘子：50/50 次调用 cos_OK。

| 步 | 内容 | 完成后 |
|---|---|---|
| S0 | 保存基准：现在的 `graph_driver.c`、weights.bin 的哈希、冒烟日志、橘子 | 有对照 |
| S1 | **按调用替换 + 步骤 ABI + bf16 权重约定**：`offload_group` 支持一组一个调用；`AIEK_STEP`；`weights=` 和 `weight_transform` | conv、linear 直接走 aiegraph，`patch_driver` 里这部分删掉 |
| S2 | **组**：多个调用、放置、删除、提前写的检查（在 graph.json 上做）；partition.json 记录组；`offload_reset` 支持整组撤回 | 有单元测试：故意构造被覆盖的 buffer、组外有读者的中间张量，必须报错 |
| S3 | **attention 匹配**：把 `map_attn` 的逻辑挪进 aiegraph，`emod.match("attention")`，从 Relay 取 scale | flash-attention 走 aiegraph，`_remove_fa_softmax` 删掉 |
| S4 | **组级 x86 校验**：子图链式参考 vs kemu | 去掉 `force=True` 和空函数补丁 |
| S5 | **keep-layout 和 temb**：接上已有的 `partition.plan_keep_layout`（现在只做规划，codegen 没接，GAP 6a9d38）；纯排列判定在 Relay 上做 | 删掉 `keep_layout_rt.py` |
| S6 | 教程改写成纯 aiegraph 调用；`build_unet.py` 只剩"生成模板 + 声明组" | 新教程，重新生成橘子 |

S2 和 S3 是主要工作量；S1、S4、S6 比较直接。

---

## 9. 未决问题

1. **matcher 的通用性**：attention 的匹配规则先只覆盖 bk-sdm-tiny 的 6 类，换模型（比如 SDXL）时再扩展。是否需要让用户直接给出 ONNX 节点列表作为兜底？
2. **步骤的状态**：中间状态在 kernel 的静态暂存区里，所以同一时刻一组只能进行一次。现在是单线程顺序执行，没有问题；以后要并行跑多个组时得重新考虑。
3. **按名替换的兼容**：现有的 ResNet 流程按函数名替换。新接口按调用替换，旧接口保留为"把这个名字的所有调用各自作为一组"的简写。
4. **x86 校验的容差**：bf16 权重和 fp32 参考之间按 cos ≥ 0.999 加 maxabs 门槛判定，具体阈值沿用教程里 runner 的设置，还是更严？
