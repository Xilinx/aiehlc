# User-kernel examples

Bring-your-own AIE kernel for TVM-deployed networks. Interface:
`include/aie_user_kernel.h` (C) and `src/frontend/tvmrelay/aiekernel/` (Python).
Full walkthrough: [doc/tutorial_userkernel.md](../../doc/tutorial_userkernel.md).

| file | what |
|---|---|
| `conv3x3_requant_ref.c` | **Reference-only** 3×3 int8 conv + requant + ReLU − 128, shape-generic (symbolic dims). Step 1 of a new kernel: on int8 ResNet-18 it matches layers 04, 09, 14, 16, 21, 23, 28 (8 calls), calibrates bit-exact against TVM on every call and leaves the logits byte-identical with all of them swapped. |
| `../../src/aietensorop/conv2dstem/conv2dstem_uk.cc` (+ `_ref.c`) | The ResNet-18 stem through the interface: real AIE kernel (spliced `conv2dstem.cc`), builds and links into `main.elf`. Lives next to `conv2dstem.cc` because aiehlc copies headers flat and splices `.cc` includes relative to the entry dir. |

```bash
export PYTHONPATH=src
python3 -m frontend.tvmrelay.aiekernel check example/userkernel/conv3x3_requant_ref.c
python3 -m frontend.tvmrelay.aiekernel check src/aietensorop/conv2dstem/conv2dstem_uk.cc
python3 -m frontend.tvmrelay.aiekernel offload src/aietensorop/conv2dstem/conv2dstem_uk.cc@1 --make
```

None of these touch a board.
