# resnetengine — whole int8 ResNet-18 on the AIE

2.81 ms per image (float image → logits), bit-identical to TVM's int8 graph.
Design, exactness argument and the measured optimization steps:
[doc/design/resnet_aie_engine.md](../../../doc/design/resnet_aie_engine.md).
How to run / extend: skill `.cursor/skills/resnetengine/SKILL.md`.

| path | what |
|---|---|
| `gen_engine.py` | TVM artifacts → engine blob + `eng_net.h`; numpy model proves bit-exactness first |
| `eng/` | the engine: `kernel.cc` (persistent AIE program), `host.cc` (driver + CPU ends), `eng_desc.h` (contract) |
| `engine_fabric.h`, `fabric/` | the fixed fabric (conv2dstem's routing.cc, partition {2,6}) |
| `build.sh` | assemble a worklocal and run hostcompile.sh → `<out>/aout/worklocal/build/host` |
| `e0/`, `e1/` | bring-up: persistent kernel on synthetic data; layer 04 alone |
