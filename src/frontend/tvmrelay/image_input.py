###############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
"""Turn a real image into the C header the board ELF classifies.

    dog.jpg  ->  ImageNet preprocess  ->  input_image.h  ->  linked into main.elf

The preprocessing is **not reimplemented here**: it calls
the example classifier's ``classify.py`` ``preprocess()``, the same function
the reference classifier uses -- resize the shorter side to 256, center-crop
224, scale to [0,1], subtract mean ``[0.485, 0.456, 0.406]``, divide by std
``[0.229, 0.224, 0.225]``, transpose to NCHW. Reimplementing it would be the
easiest possible way to make the board and the CPU reference disagree for a
reason that has nothing to do with the compiler, so they share one
implementation.

The result is emitted as a ``static const float[150528]`` in ``input_image.h``.
Float, not int8: this path is fp32 end to end (stage 3 is gated on LLVM), so
quantizing here would only add error the model was not built for.

``imagenet_labels.h`` rides along so the ELF can print "Samoyed" rather than
"class 258" -- a class index alone does not tell you whether the run was
correct.

Both headers are plain data with no dependencies, so they compile for the
board and for the x86 CPU check from the same source.
"""

from __future__ import annotations

import os
import sys
import textwrap
from pathlib import Path

__all__ = ["preprocess_image", "write_image_header", "write_labels_header",
           "load_labels", "classify_reference"]

#: The directory holding ``classify.py`` (preprocessing + label loading). It is
#: not a package, so it goes on sys.path directly. ``$AIEHLC_CLASSIFY_DIR``
#: overrides; otherwise the one ``example/model/*/classify.py`` in the repo.
_EXAMPLE_MODELS = Path(__file__).resolve().parents[3] / "example" / "model"


def _classify_dir() -> Path:
    env = os.environ.get("AIEHLC_CLASSIFY_DIR")
    if env:
        return Path(env)
    hits = sorted(_EXAMPLE_MODELS.glob("*/classify.py"))
    if not hits:
        raise FileNotFoundError(f"no {_EXAMPLE_MODELS}/*/classify.py "
                                f"(set AIEHLC_CLASSIFY_DIR)")
    return hits[0].parent


def _classify_mod():
    """Import ``classify`` from :func:`_classify_dir`. Raises if absent."""
    path = str(_classify_dir())
    if path not in sys.path:
        sys.path.insert(0, path)
    import classify  # noqa: F401

    return classify


def preprocess_image(image=None, verbose: bool = True):
    """``(tensor, image_path)`` -- NCHW float32 ``[1,3,224,224]``, ImageNet norm.

    *image* may be a local path, a URL, or None for the standard pytorch/hub
    dog photo. Needs torch + PIL (``classify.preprocess`` returns a torch
    tensor) and, for a URL or the default, network access on first use; the
    download is cached by ``classify.resolve``.
    """
    if image is not None and str(image).endswith((".bin", ".npy")):
        # multi-input graphs: a non-image input, given as the raw float32
        # tensor itself (little-endian .bin, or .npy). No preprocessing.
        import numpy as np
        from pathlib import Path as _P
        path = _P(image).resolve()
        tensor = (np.load(path) if str(path).endswith(".npy")
                  else np.fromfile(path, dtype="<f4")).astype(np.float32).ravel()
        if verbose:
            print(f"  [input] raw float32 tensor {path} ({tensor.size} elems)")
        return tensor, path
    classify = _classify_mod()
    path = classify.resolve(image, classify.DEFAULT_IMAGE_URL, "input_image")
    tensor = classify.preprocess(path).numpy()
    if verbose:
        print(f"  [image] {path} -> {tuple(tensor.shape)} "
              f"range [{tensor.min():.3f}, {tensor.max():.3f}]")
    return tensor, path


def load_labels(verbose: bool = True) -> list:
    """The 1000 ImageNet class names, downloaded and cached on first use."""
    classify = _classify_mod()
    path = classify.resolve(None, classify.DEFAULT_LABELS_URL,
                            "imagenet_classes.txt")
    return classify.load_labels(path)


def classify_reference(tensor, onnx_path, labels=None, topk: int = 5) -> list:
    """Run *tensor* through *onnx_path* with onnxruntime. Returns the top-k.

    This is the independent oracle the board result is checked against -- a
    second implementation of the same network, not a second call into the one
    being tested. Each entry is ``(rank, class_index, label, logit)``.
    """
    import numpy as np
    import onnxruntime as ort

    sess = ort.InferenceSession(str(onnx_path),
                                providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    logits = sess.run(None, {name: tensor})[0].ravel()
    order = np.argsort(logits)[-topk:][::-1]
    return [(rank, int(i), (labels[i] if labels else str(i)), float(logits[i]))
            for rank, i in enumerate(order, 1)]


# ═══════════════════════════════════════════════════════════════════════════
#  Header emission
# ═══════════════════════════════════════════════════════════════════════════

_IMAGE_H_HEAD = """\
/* Generated by src/frontend/tvmrelay/image_input.py -- do not edit.
 *
 * {source}
 * ImageNet preprocessing (example classifier classify.py:preprocess):
 * resize shorter side to 256, center-crop 224, /255, subtract
 * mean [0.485, 0.456, 0.406], divide std [0.229, 0.224, 0.225], NCHW.
 *
 * Layout is NCHW float32 [1, 3, 224, 224] -- exactly what graph_input()
 * expects, so main.c copies it in without rearranging.
 */
#ifndef TVMRELAY_INPUT_IMAGE_H
#define TVMRELAY_INPUT_IMAGE_H

#define INPUT_IMAGE_ELEMS {elems}

static const float input_image[INPUT_IMAGE_ELEMS] = {{
"""

_IMAGE_H_TAIL = """\
};

#endif  /* TVMRELAY_INPUT_IMAGE_H */
"""


def write_image_header(tensor, out_path: Path, source: str = "",
                       verbose: bool = True) -> Path:
    """Emit *tensor* as ``static const float input_image[]``. Returns the path.

    ``repr`` of a numpy float32 round-trips exactly (Python prints the shortest
    string that reads back to the same value), so the header is bit-identical
    to what the CPU reference sees -- the two cannot drift through formatting.
    """
    flat = tensor.ravel()
    body = ", ".join(f"{v!r}f" for v in flat.tolist())
    text = (_IMAGE_H_HEAD.format(source=source or "generated input",
                                 elems=flat.size)
            + textwrap.fill(body, width=78, initial_indent="    ",
                            subsequent_indent="    ")
            + "\n" + _IMAGE_H_TAIL)
    out_path.write_text(text)
    if verbose:
        print(f"  [image] {out_path.name} ({flat.size} floats, "
              f"{out_path.stat().st_size / 1e6:.1f} MB)")
    return out_path


_LABELS_H_HEAD = """\
/* Generated by src/frontend/tvmrelay/image_input.py -- do not edit.
 *
 * ImageNet class names, so the ELF prints "Samoyed" and not "class 258".
 */
#ifndef TVMRELAY_IMAGENET_LABELS_H
#define TVMRELAY_IMAGENET_LABELS_H

#define IMAGENET_NUM_CLASSES {n}

static const char *const imagenet_labels[IMAGENET_NUM_CLASSES] = {{
"""


def write_labels_header(labels: list, out_path: Path,
                        verbose: bool = True) -> Path:
    """Emit *labels* as ``static const char *const imagenet_labels[]``."""
    lines = [_LABELS_H_HEAD.format(n=len(labels))]
    for name in labels:
        escaped = name.replace("\\", "\\\\").replace('"', '\\"')
        lines.append(f'    "{escaped}",\n')
    lines.append("};\n\n#endif  /* TVMRELAY_IMAGENET_LABELS_H */\n")
    out_path.write_text("".join(lines))
    if verbose:
        print(f"  [image] {out_path.name} ({len(labels)} classes)")
    return out_path
