# Custom PyTorch operators through torch.export → aiednn eager mode

This example shows every way a custom PyTorch operator can be handled when you
compile a model for the AI Engine with `torch.export` + eager mode.

The model is always the same skeleton:

```
input [32, 256]  →  Linear(256→128)  →  custom op  →  Linear(128→64)  →  output [32, 64]
```

The two Linears become AIE GEMM instances.  What happens to the custom op
depends on the case.

---

## Case A — `torch.library.custom_op` with a Python body (auto-inline)

### How it works

When you define an op with `torch.library.custom_op`, torch keeps the Python
implementation in its internal `OPDEFS` registry.  `export_inlined()` reads
that registry, finds the body, and re-exports the model with the op replaced by
its ATen IR.  The opaque node disappears before the compiler ever sees it.

```python
@torch.library.custom_op("demo::softclip_a", mutates_args=())
def softclip_a(t: torch.Tensor) -> torch.Tensor:
    return torch.tanh(t) * 2.0          # a pure-PyTorch body

@softclip_a.register_fake
def _(t):
    return torch.empty_like(t)

ep = dnn_export.export_inlined(model, [x])
# -- the graph now has only the two Linear ops; softclip_a is gone --
compile_exported(ep, x, eager="library")
```

### What the compiler sees after inlining

```
call [0] aten.mm          → 32×128  (Linear 1, GEMM on AIE)
call [1] aten.tanh        → 32×128  (inlined body, stays on host)
call [2] aten.mul_scalar  → 32×128  (inlined body, stays on host)
call [3] aten.mm          → 32×64   (Linear 2, GEMM on AIE)
```

The `tanh` and `mul` are small elementwise ops that the library puts on `ref:`
(the A72).  The two GEMMs are on the AIE.

### When to use

Your op is implemented in PyTorch ops.  It may or may not be a bottleneck; you
just want it out of the way so the surrounding GEMMs see the AIE.

---

## Case B — older-style registration, body passed explicitly

### How it works

If the op was registered against a backend key (`lib.impl("CPU", fn)`) torch
does not keep the body in `OPDEFS`.  Pass it yourself:

```python
_lib = torch.library.Library("demo_b", "DEF")
_lib.define("softclip_b(Tensor t) -> Tensor")
_lib.impl("softclip_b", lambda t: torch.tanh(t) * 2.0, "CPU")

ep = dnn_export.export_inlined(model, [x],
         bodies={"demo_b::softclip_b": lambda t: torch.tanh(t) * 2.0})
```

The body is registered as `CompositeImplicitAutograd` for the second
`torch.export` call only, then deregistered.  The result is identical to
case A.

### When to use

An op already in production, registered the old way, whose body is known.  No
source changes required; just hand the body to `export_inlined`.

---

## Case C — opaque custom op, C implementation on the host

### How it works

No Python body is available (or you deliberately keep the op opaque).  The
exported graph retains one `call_function` node for it.  The compiler maps this
to an `aiednn_custom` primitive — a function pointer called on the A72 between
the two AIE GEMMs.

You supply the C file; the compiler copies it into the build and checks the
symbol is present before linking.

```python
# plain export — the op stays opaque
ep = torch.export.export(model, (x,))

# compile with the C file for the custom op
compile_exported(ep, x, eager="library",
                 custom_sources=["softclip_c.c"])
```

### The C function

The name is derived from the op's qualified name:

```
demo::softclip_c.default  →  aiednn_custom_demo_softclip_c_default
```

Rule: `"aiednn_custom_" + name.replace("::", "_").replace(".", "_")` with any
remaining non-word characters replaced by `_`.  You can derive it in Python:

```python
from aiegraph.dnn_torch import custom_c_symbol
print(custom_c_symbol("demo::softclip_c.default"))
# aiednn_custom_demo_softclip_c_default
```

The signature is always:

```c
int aiednn_custom_demo_softclip_c_default(
        int nsrc,
        const aiednn_memory_desc_t *srcs,   /* [nsrc] input descriptors  */
        const void *const *src_data,         /* [nsrc] input pointers     */
        const aiednn_memory_desc_t *dst,     /* output descriptor         */
        void *dst_data,                      /* output pointer            */
        void *ctx);                          /* NULL unless you passed it */
```

The runtime has already allocated and cache-synced the input and output buffers
before calling you.  Return 0 for success, non-zero to signal an error.  The
`aiednn_memory_desc_t` fields give you the shape, dtype, and strides if you
need them (`aiednn_md_nelems(dst)` is the total element count).

The C for clip-to-[-1,1]:

```c
int aiednn_custom_demo_softclip_c_default(
        int nsrc, const aiednn_memory_desc_t *srcs, const void *const *src_data,
        const aiednn_memory_desc_t *dst, void *dst_data, void *ctx) {
    (void)srcs; (void)ctx;
    if (nsrc != 1) return -1;
    const float *x = (const float *)src_data[0];
    float *y = (float *)dst_data;
    int64_t n = aiednn_md_nelems(dst);
    for (int64_t i = 0; i < n; i++) {
        float v = x[i];
        y[i] = v < -1.0f ? -1.0f : v > 1.0f ? 1.0f : v;
    }
    return 0;
}
```

### What the compiler sees

```
call [0] aten.mm          → 32×128  (Linear 1, GEMM on AIE)
call [1] demo::softclip_c → 32×128  (aiednn_custom on the A72)
call [2] aten.mm          → 32×64   (Linear 2, GEMM on AIE)
```

The GEMMs run on the AIE; the custom op runs on the A72 between them.

### When to use

The op calls a library with no torch equivalent (a lookup table, a hardware
driver, a codec), or you want to control the host-side implementation exactly
(e.g., NEON intrinsics).

---

## Running the example

```bash
# workstation emulation (all three cases)
PYTHONPATH=src python example/aiednn/custom_pytorch_op/run.py

# one case at a time
PYTHONPATH=src python example/aiednn/custom_pytorch_op/run.py --case a
PYTHONPATH=src python example/aiednn/custom_pytorch_op/run.py --case b
PYTHONPATH=src python example/aiednn/custom_pytorch_op/run.py --case c

# board run (VEK280)
PYTHONPATH=src python example/aiednn/custom_pytorch_op/run.py \
    --target vek280 --board galerina13 \
    --vitis-settings /proj/xbuilds/.../Vitis/settings64.sh
```

---

## Decision tree

```
Is the op implemented in PyTorch ops?
├── yes → export_inlined()
│         ├── torch.library.custom_op?  →  Case A (body auto-found)
│         └── older lib.impl("CPU")?    →  Case B (pass body=...)
└── no  → plain torch.export.export() + custom_sources=[file.c]  →  Case C
```

If you are not sure: try `export_inlined()` without `bodies=`; if the op is
still present in the graph after export (check for `is_custom_op` nodes) it has
no Python body and you need case B or C.
