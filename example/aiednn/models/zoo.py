###############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
"""Models the aiednn model tests run: torchvision classifiers and small custom networks.

    zoo.build("mobilenet_v2", pretrained=True) -> (model, input)
"""

from __future__ import annotations

import types
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[3]
IMAGE = REPO / "worklocal" / "aiednn" / "resnet18_data" / "r18_image.bin"   # gen_resnet18.py

TORCHVISION = ["resnet18", "mobilenet_v2", "mobilenet_v3_small", "squeezenet1_1", "efficientnet_b0",
               "regnet_x_400mf", "densenet121", "convnext_tiny"]


class TinyViT(nn.Module):
    """Tiny ViT: patch embedding, [CLS] token, pre-norm transformer blocks, head."""

    def __init__(self, img=32, patch=4, dim=64, heads=4, depth=2, classes=10):
        super().__init__()
        self.heads, self.dim = heads, dim
        self.embed = nn.Conv2d(3, dim, patch, patch)
        n = (img // patch) ** 2
        self.pos = nn.Parameter(torch.randn(1, n, dim) * 0.02)
        self.blocks = nn.ModuleList([nn.ModuleDict(dict(
            n1=nn.LayerNorm(dim), q=nn.Linear(dim, dim), k=nn.Linear(dim, dim), v=nn.Linear(dim, dim),
            o=nn.Linear(dim, dim), n2=nn.LayerNorm(dim), f1=nn.Linear(dim, 2 * dim), f2=nn.Linear(2 * dim, dim)))
            for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, classes)

    def heads_split(self, t, n):
        return t.reshape(1, n, self.heads, self.dim // self.heads).permute(0, 2, 1, 3)

    def forward(self, x):
        t = self.embed(x).flatten(2).transpose(1, 2) + self.pos          # [1, N, D]
        n = t.shape[1]
        for b in self.blocks:
            h = b["n1"](t)
            q, k, v = (self.heads_split(b[w](h), n) for w in ("q", "k", "v"))
            a = F.scaled_dot_product_attention(q, k, v)
            t = t + b["o"](a.permute(0, 2, 1, 3).reshape(1, n, self.dim))
            t = t + b["f2"](F.gelu(b["f1"](b["n2"](t))))
        return self.head(self.norm(t).mean(dim=1))


class TinyUNet(nn.Module):
    """Tiny U-Net: conv/maxpool encoder, transposed-conv / bilinear decoder with skips, sigmoid head."""

    def __init__(self, c=16):
        super().__init__()
        def blk(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, 1, 1), nn.BatchNorm2d(o), nn.ReLU())
        self.e1, self.e2, self.mid = blk(3, c), blk(c, 2 * c), blk(2 * c, 4 * c)
        self.up2 = nn.ConvTranspose2d(4 * c, 2 * c, 2, 2)
        self.d2 = blk(4 * c, 2 * c)
        self.d1 = blk(3 * c, c)
        self.out = nn.Conv2d(c, 3, 1)

    def forward(self, x):
        a = self.e1(x)
        b = self.e2(F.max_pool2d(a, 2))
        m = self.mid(F.max_pool2d(b, 2))
        d2 = self.d2(torch.cat([self.up2(m), b], 1))
        d1 = self.d1(torch.cat([F.interpolate(d2, scale_factor=2, mode="bilinear", align_corners=False), a], 1))
        return torch.sigmoid(self.out(d1))


class MixerHead(nn.Module):
    """grouped / dilated convs, PReLU, GroupNorm, LayerNorm, avg pooling, softmax"""

    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(3, 32, 3, 2, 1)
        self.g1 = nn.GroupNorm(4, 32)
        self.act = nn.PReLU(32)
        self.c2 = nn.Conv2d(32, 64, 3, 1, 2, dilation=2, groups=4)
        self.pool = nn.AvgPool2d(3, 2, 1, count_include_pad=False)
        self.ln = nn.LayerNorm(64)
        self.fc = nn.Linear(64, 10)

    def forward(self, x):
        y = self.pool(F.silu(self.c2(self.act(self.g1(self.c1(x))))))
        y = self.ln(y.mean(dim=(2, 3)))
        return F.softmax(self.fc(y), dim=-1)


CUSTOM = {"tiny_vit": (TinyViT, (1, 3, 32, 32)), "tiny_unet": (TinyUNet, (1, 3, 32, 32)),
          "mixer_head": (MixerHead, (1, 3, 32, 32))}


# ── benchmark models: not in the default run_models list; transformers take embeddings, not token ids ──

class BertLayer(nn.Module):
    """post-norm encoder layer: fused-QKV-free self attention with an additive mask, GELU (erf) MLP"""

    def __init__(self, d, heads, ff):
        super().__init__()
        self.h = heads
        self.q, self.k, self.v, self.o = (nn.Linear(d, d) for _ in range(4))
        self.n1, self.n2 = nn.LayerNorm(d, eps=1e-12), nn.LayerNorm(d, eps=1e-12)
        self.f1, self.f2 = nn.Linear(d, ff), nn.Linear(ff, d)

    def split(self, t):
        b, s, d = t.shape
        return t.reshape(b, s, self.h, d // self.h).permute(0, 2, 1, 3)

    def forward(self, x, mask):
        b, s, d = x.shape
        a = F.scaled_dot_product_attention(self.split(self.q(x)), self.split(self.k(x)), self.split(self.v(x)),
                                           attn_mask=mask)
        x = self.n1(x + self.o(a.permute(0, 2, 1, 3).reshape(b, s, d)))
        return self.n2(x + self.f2(F.gelu(self.f1(x))))


class BertBase(nn.Module):
    """BERT-base encoder (12 x 768, 12 heads, FFN 3072) on [1, S, 768] embeddings + a padding mask"""

    def __init__(self, layers=12, d=768, heads=12, ff=3072, seq=128, valid=100):
        super().__init__()
        self.emb_norm = nn.LayerNorm(d, eps=1e-12)
        self.layers = nn.ModuleList([BertLayer(d, heads, ff) for _ in range(layers)])
        m = torch.zeros(1, 1, 1, seq)
        m[..., valid:] = -10000.0                      # the last tokens are padding
        self.register_buffer("mask", m)
        self.pool = nn.Linear(d, d)

    def forward(self, x):
        x = self.emb_norm(x)
        for layer in self.layers:
            x = layer(x, self.mask)
        return torch.tanh(self.pool(x[:, 0]))


class GPT2Block(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.h = heads
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.proj = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.fc, self.out = nn.Linear(d, 4 * d), nn.Linear(4 * d, d)

    def forward(self, x):
        b, s, d = x.shape
        q, k, v = self.qkv(self.ln1(x)).split(d, dim=2)
        sh = lambda t: t.reshape(b, s, self.h, d // self.h).permute(0, 2, 1, 3)
        a = F.scaled_dot_product_attention(sh(q), sh(k), sh(v), is_causal=True)
        x = x + self.proj(a.permute(0, 2, 1, 3).reshape(b, s, d))
        return x + self.out(F.gelu(self.fc(self.ln2(x)), approximate="tanh"))


class GPT2Small(nn.Module):
    """GPT-2 small (12 x 768, 12 heads), pre-norm, causal attention, the prefill pass on [1, S, 768]"""

    def __init__(self, layers=12, d=768, heads=12):
        super().__init__()
        self.blocks = nn.ModuleList([GPT2Block(d, heads) for _ in range(layers)])
        self.ln_f = nn.LayerNorm(d)

    def forward(self, x):
        for b in self.blocks:
            x = b(x)
        return self.ln_f(x)


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.eps, self.weight = eps, nn.Parameter(torch.ones(d))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


def rope(t, cos, sin):
    """rotary embedding, the half-split (LLaMA / HF) form"""
    h = t.shape[-1] // 2
    return t * cos + torch.cat([-t[..., h:], t[..., :h]], dim=-1) * sin


class LlamaBlock(nn.Module):
    def __init__(self, d, heads, kv_heads, ff):
        super().__init__()
        self.h, self.kvh, self.hd = heads, kv_heads, d // heads
        self.n1, self.n2 = RMSNorm(d), RMSNorm(d)
        self.q = nn.Linear(d, d, bias=False)
        self.k = nn.Linear(d, kv_heads * self.hd, bias=False)
        self.v = nn.Linear(d, kv_heads * self.hd, bias=False)
        self.o = nn.Linear(d, d, bias=False)
        self.gate, self.up = nn.Linear(d, ff, bias=False), nn.Linear(d, ff, bias=False)
        self.down = nn.Linear(ff, d, bias=False)

    def forward(self, x, cos, sin):
        b, s, d = x.shape
        h = self.n1(x)
        q = rope(self.q(h).reshape(b, s, self.h, self.hd).permute(0, 2, 1, 3), cos, sin)
        k = rope(self.k(h).reshape(b, s, self.kvh, self.hd).permute(0, 2, 1, 3), cos, sin)
        v = self.v(h).reshape(b, s, self.kvh, self.hd).permute(0, 2, 1, 3)
        rep = self.h // self.kvh                        # GQA: each K / V head serves rep query heads
        k, v = k.repeat_interleave(rep, dim=1), v.repeat_interleave(rep, dim=1)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.o(a.permute(0, 2, 1, 3).reshape(b, s, d))
        h = self.n2(x)
        return x + self.down(F.silu(self.gate(h)) * self.up(h))   # SwiGLU gated MLP


class TinyLlama(nn.Module):
    """LLaMA-style decoder (TinyLlama's shape, scaled down): RMSNorm, RoPE, GQA (8 query / 2 KV heads),
    SwiGLU; prefill on [1, S, 512]"""

    def __init__(self, layers=4, d=512, heads=8, kv_heads=2, ff=1408, seq=128):
        super().__init__()
        self.blocks = nn.ModuleList([LlamaBlock(d, heads, kv_heads, ff) for _ in range(layers)])
        self.norm = RMSNorm(d)
        hd = d // heads
        inv = 1.0 / (10000 ** (torch.arange(0, hd, 2).float() / hd))
        f = torch.outer(torch.arange(seq).float(), inv)
        emb = torch.cat([f, f], dim=-1)
        self.register_buffer("cos", emb.cos()[None, None])
        self.register_buffer("sin", emb.sin()[None, None])

    def forward(self, x):
        for b in self.blocks:
            x = b(x, self.cos, self.sin)
        return self.norm(x)


YOLOV8N_PT = REPO / "worklocal" / "models" / "yolov8n.pt"
YOLOV8N_URL = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt"


def _yolov8n():
    """yolov8n with the released COCO weights (random weights give a degenerate model)."""
    import ultralytics.nn.tasks  # noqa: F401 -- unpickles the model
    if not YOLOV8N_PT.is_file():
        import urllib.request
        YOLOV8N_PT.parent.mkdir(parents=True, exist_ok=True)
        try:
            urllib.request.urlretrieve(YOLOV8N_URL, YOLOV8N_PT)
        except OSError as e:
            raise SystemExit(f"yolov8n: cannot download {YOLOV8N_URL} ({e}); put the file at {YOLOV8N_PT}")
    m = torch.load(YOLOV8N_PT, map_location="cpu", weights_only=False)["model"].float()
    m.model[-1].export = True                          # the head returns one tensor (no NMS inside)
    from ultralytics.nn.modules.block import C2f
    for mod in m.modules():
        if isinstance(mod, C2f):
            mod.forward = types.MethodType(_c2f_forward, mod)
    return m


def _c2f_forward(self, x):
    """C2f.forward as a plain loop: Dynamo mis-traces upstream's y.extend(generator) for n >= 2."""
    y = list(self.cv1(x).chunk(2, 1))
    for m in self.m:
        y.append(m(y[-1]))
    return self.cv2(torch.cat(y, 1))


def _yolo_image(shape) -> torch.Tensor:
    """ultralytics' bundled bus.jpg, RGB in [0, 1], resized (not letterboxed) to *shape*'s H x W"""
    import cv2
    import ultralytics
    img = cv2.imread(str(Path(ultralytics.__file__).parent / "assets" / "bus.jpg"))
    img = cv2.resize(img, (shape[3], shape[2]), interpolation=cv2.INTER_AREA)[:, :, ::-1]
    return torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1))).float().div(255).unsqueeze(0)


BENCH = {"resnet50": None, "vit_b_16": None,           # torchvision, ImageNet weights
         "bert_base": (BertBase, (1, 128, 768)), "gpt2_small": (GPT2Small, (1, 128, 768)),
         "tiny_llama": (TinyLlama, (1, 128, 512)), "yolov8n": (_yolov8n, (1, 3, 320, 320))}


def image_input() -> torch.Tensor:
    if not IMAGE.is_file():
        raise SystemExit(f"{IMAGE} missing: run example/aiednn/resnet18/gen_resnet18.py")
    return torch.from_numpy(np.fromfile(IMAGE, np.float32).reshape(1, 3, 224, 224))


def build(name: str, pretrained: bool = True, seed: int = 0):
    """(eval model, example input); pretrained weights and the real image when *pretrained*."""
    torch.manual_seed(seed)
    if BENCH.get(name):                                 # transformers: random weights
        ctor, shape = BENCH[name]
        m = ctor().eval()
        if name != "yolov8n":
            return m, torch.randn(shape)
        x = _yolo_image(shape)
        with torch.no_grad():
            n = m(x).shape[-1]                         # [1, 4 + 80, anchors]: box rows first
        m.top1_range = (4 * n, 84 * n, n)              # top-1 = class of the strongest detection
        return m, x
    if name in CUSTOM:
        cls, shape = CUSTOM[name]
        m = cls().eval()
        for mod in m.modules():                      # non-trivial BN statistics
            if isinstance(mod, nn.BatchNorm2d):
                mod.running_mean.uniform_(-0.2, 0.2)
                mod.running_var.uniform_(0.5, 2.0)
        return m, torch.randn(shape)
    import torchvision.models as tvm
    m = getattr(tvm, name)(weights="DEFAULT" if pretrained else None).eval()
    return m, image_input() if pretrained else torch.randn(1, 3, 224, 224)
