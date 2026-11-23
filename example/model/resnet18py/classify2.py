###############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
import argparse
import ast
import os
import shutil
import urllib.request
from urllib.parse import urlparse

import numpy as np
import torch
from PIL import Image

from resnet18 import resnet18

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.environ.get("RESNET18PY_CACHE", os.path.expanduser("~/.cache/resnet18py"))

# Everything is fetched from the network by default.
DEFAULT_IMAGE_URL = "https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg"
DEFAULT_WEIGHTS_URL = (
    "https://github.com/onnx/models/raw/main/validated/vision/"
    "classification/resnet/model/resnet18-v2-7.onnx"
)
DEFAULT_LABELS_URL = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _is_url(s):
    return s.startswith(("http://", "https://"))


def _download(url, dest):
    """Download `url` to `dest`, skipping if already cached."""
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        print(f"[cache]    {dest}")
        return dest
    print(f"[download] {url}")
    tmp = dest + ".part"
    with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f)
    os.replace(tmp, dest)
    print(f"[saved]    {dest} ({os.path.getsize(dest):,} bytes)")
    return dest


def resolve(src, default_url, cache_name):
    """Return a local path for `src` (a URL, a local path, or None -> default)."""
    src = src if src is not None else default_url
    if _is_url(src):
        name = os.path.basename(urlparse(src).path) or cache_name
        return _download(src, os.path.join(CACHE, name))
    return src


def load_labels(path):
    """Parse ImageNet labels: either a `{0: '...', ...}` dict or one name per line."""
    text = open(path).read()
    if text.lstrip().startswith("{"):
        table = ast.literal_eval(text)
        return [table[i] for i in range(len(table))]
    labels = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        # Drop a leading WordNet synset id ("n01440764 tench, ..." -> "tench, ...").
        parts = line.split(maxsplit=1)
        if len(parts) == 2 and parts[0].startswith("n") and parts[0][1:].isdigit():
            line = parts[1]
        labels.append(line)
    return labels

def preprocess(path, resize=256, crop=224):
    img = Image.open(path).convert("RGB")
    w, h = img.size
    if w < h:
        new_w, new_h = resize, round(h*resize/w)
    else:
        new_w, new_h = round(w * resize / h), resize

    img = img.resize((new_w, new_h), Image.BILINEAR)

    left = (new_w - crop) // 2
    top = (new_h - crop) // 2
    img = img.crop((left, top, left + crop, top + crop))

    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - MEAN)/STD
    arr = np.transpose(arr,(2,0,1))
    return torch.from_numpy(arr).unsqueeze(0)

def main():
    ap = argparse.ArgumentParser(description="Resnet18 classifier")
    ap.add_argument("image", nargs="?", default=None, help="Path or URL to an image (default: %(default)s)")
    ap.add_argument("--weights", default=None, help="Path or URL to ONNX weights (default: %(default)s)")
    ap.add_argument("--labels", default=None,
                    help="imagenet labels URL or path (default: download)")
    ap.add_argument("--topk", type=int, default=5)
    args = ap.parse_args()

    image_path = resolve(args.image, DEFAULT_IMAGE_URL, "input_image")
    weights_path = resolve(args.weights, DEFAULT_WEIGHTS_URL, "resnet18-v2-7.onnx")
    labels_path = resolve(args.labels, DEFAULT_LABELS_URL, "imagenet_classes.txt")

    labels = load_labels(labels_path)
    model = resnet18(onnx_path=weights_path)  

    x = preprocess(image_path)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]

    topv, topi = probs.topk(args.topk)
    print(f"image: {image_path}")
    print(f"model: {weights_path}")
    print(f"top-{args.topk} predictions:")
    for rank, (p, i) in enumerate(zip(topv.tolist(), topi.tolist()), 1):
        print(f"  {rank}. {labels[i]:<55s} {p*100:6.2f}%  (class {i})")

if __name__ == "__main__":
    main()