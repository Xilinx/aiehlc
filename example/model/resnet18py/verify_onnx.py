###############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
"""Cross-check the from-scratch PyTorch ResNet-18 against the ONNX reference.

Runs the same preprocessed input through onnxruntime and the PyTorch model and
compares logits. Confirms the hand-written implementation is faithful.
"""

import numpy as np
import onnxruntime as ort
import torch

from classify import DEFAULT_IMAGE_URL, DEFAULT_WEIGHTS_URL, preprocess, resolve
from resnet18 import resnet18


def main():
    image_path = resolve(None, DEFAULT_IMAGE_URL, "input_image")
    weights_path = resolve(None, DEFAULT_WEIGHTS_URL, "resnet18-v2-7.onnx")

    x = preprocess(image_path)                    # [1,3,224,224]

    model = resnet18(onnx_path=weights_path)
    with torch.no_grad():
        torch_out = model(x).numpy()[0]

    sess = ort.InferenceSession(weights_path, providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    onnx_out = sess.run(None, {in_name: x.numpy()})[0][0]

    max_abs = np.max(np.abs(torch_out - onnx_out))
    print(f"max |logit diff|        : {max_abs:.3e}")
    print(f"pytorch argmax          : {int(torch_out.argmax())}")
    print(f"onnxruntime argmax      : {int(onnx_out.argmax())}")
    ok = max_abs < 1e-3 and torch_out.argmax() == onnx_out.argmax()
    print("RESULT                  :", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
