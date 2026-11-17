# Tutorials

This folder holds the tutorials for **aiehlc**, the AIE high-level compiler.
aiehlc compiles one C++ file, which contains host code and an AIE kernel, into
an ELF that runs on an AMD Versal AI Engine board. The folder also holds the
prebuilt hardware images the tutorials run on.

## Where to start

| Tutorial | What it teaches | Start here if |
|---|---|---|
| **[aieprogram](aieprogram/README.md)** | The **kernel-only programming model**. One `__global__` kernel says, in tensor coordinates, which block each core loads and stores; the compiler derives the DMA, routing and buffers. Chapters go by operator family (contraction, elementwise, reduction, data movement, indexing), then optimize GEMM step by step (scheduling options, vector instructions). | you are new to programming the AIE array with aiehlc |
| **[aiednn](aiednn/howto/templatekernel.md)** | **Writing an AIEDNN kernel.** AIEDNN is a library of fast AIE kernels for ML offload. The tutorial covers how to write a kernel that is fast for a whole layer, why each design decision is made and what it trades off, how to integrate the kernel into the AIEDNN library, and how to add a custom op. Example: a bf16 GEMM written as a C++ template. | you know the programming model and want a library-grade ML kernel |
| [tutorial.md](tutorial.md) | The **low-level driver API**: a single kernel loaded, routed, fed and run by hand through `xaiengine` calls ([example.cpp](example.cpp)). | you need to see what the compiler generates for you |

Read aieprogram before aiednn: the AIEDNN tutorial assumes `load_buffer`,
`program_id` and the `{.depth}` / `{.resident}` / `{.tile}` options.

## Other contents

| Path | What |
|---|---|
| [hw/](hw) | Prebuilt hardware designs (PDI) the tutorials target: `vck190.pdi`, `vek280.pdi`, `vek385-reva/`, `vek385-revb/` |
| [bootbin/](bootbin) | `bootgen.bif` + `build.sh` to package a PDI with PLM, PSM firmware, device tree, ATF and U-Boot into `BOOT.bin` |
