# Custom PyTorch operators through torch.export → aiednn eager mode

This example shows every way a custom PyTorch operator can be handled when you compile a model for the AI
Engine with `torch.export` + eager mode ([`run.py`](run.py)). The model is always the same skeleton:

```
input [32, 256]  →  Linear(256→128)  →  custom op  →  Linear(128→64)  →  output [32, 64]
```

The two Linears become AIE GEMM instances. What happens to the custom op depends on the case.

`dnn_export.is_custom_op` decides what counts as a custom op: a call of `torch.ops.<ns>.<op>` outside
PyTorch's own namespaces (`aten`, `prims`, the quantized ones).

---

## Case A — a Python body, found automatically (inlined)

The op is written in torch ops. `torch.library.custom_op` keeps that body, and
`dnn_export.export_inlined()` finds it: it registers the body as the op's `CompositeImplicitAutograd`
kernel for one more `torch.export`, so the op turns into ordinary ATen ops, then removes the registration.

```python
@torch.library.custom_op("custom_op_demo_a::softclip_a", mutates_args=())
def softclip_a(t: torch.Tensor) -> torch.Tensor:
    return torch.tanh(t) * 2.0

@softclip_a.register_fake
def _(t):
    return torch.empty_like(t)

ep = dnn_export.export_inlined(model, [x])      # softclip_a is gone: tanh and mul in its place
compile_exported(ep, x, eager="library")
```

`torch.library.custom_op` needs a torch 2.4 release or newer. On an older torch, `run.py` registers the
same body as a `CompositeImplicitAutograd` kernel instead, which a plain `torch.export.export` traces
through, and the compiled graph is the same.

After inlining, the compiler sees ordinary layers:

```
aten.mm    → 32×128  (Linear 1, GEMM on the AIE)
aten.tanh  → 32×128  (the inlined body)
aten.mul   → 32×128  (the inlined body)
aten.mm    → 32×64   (Linear 2, GEMM on the AIE)
```

Use this when the op is implemented in torch ops: it then lowers like any other layer.

---

## Case B — older-style registration, body passed explicitly

An op registered for a backend key (`lib.impl(..., "CPU")`) does not keep a body torch can hand back, so
pass it yourself:

```python
_lib = torch.library.Library("custom_op_demo_b", "DEF")
_lib.define("softclip_b(Tensor t) -> Tensor")
_lib.impl("softclip_b", lambda t: torch.tanh(t) * 2.0, "CPU")

ep = dnn_export.export_inlined(model, [x],
         bodies={"custom_op_demo_b::softclip_b": lambda t: torch.tanh(t) * 2.0})
```

The body is registered as `CompositeImplicitAutograd` for that second `torch.export` only. The result is
the same as case A. Use this for an op already registered the old way whose body you know.

---

## Case C — an opaque op, a C function on the host

No Python body is available (or you keep the op opaque on purpose). The exported graph keeps one node for
it, which the compiler lowers to an `aiednn_custom` primitive: your C function, called on the A72 between
the two AIE GEMMs. Pass the C file with `custom_sources=` (flow 2: `--custom-c`):

```python
ep = torch.export.export(model, (x,))           # plain export: the op stays opaque
compile_exported(ep, x, eager="library", custom_sources=["softclip_c.c"])
```

The build copies the file next to the generated app and checks that it defines the function before linking.

### The C function

Its name comes from the op's qualified name (`dnn_torch.custom_c_symbol`): `aiednn_custom_` followed by
the name with `::` and `.` replaced by `_` (any other non-word character too):

```
custom_op_demo_c::softclip_c.default  →  aiednn_custom_custom_op_demo_c_softclip_c_default
```

```python
from aiegraph.dnn_torch import custom_c_symbol
custom_c_symbol("custom_op_demo_c::softclip_c.default")
```

Its signature is `aiednn_custom_fn_t` (`include/aiednn.h`). The runtime has allocated and cache-synced every
buffer before the call; return 0 on success. [`softclip_c.c`](softclip_c.c), clip to [-1, 1]:

```c
#include "aiednn.h"
#include <stdint.h>

int aiednn_custom_custom_op_demo_c_softclip_c_default(int nsrc, const aiednn_memory_desc_t *srcs,
                                                      const void *const *src_data, const aiednn_memory_desc_t *dst,
                                                      void *dst_data, void *ctx) {
    (void)srcs;
    (void)ctx;
    if (nsrc != 1)
        return -1;
    const float *x = (const float *)src_data[0];
    float *y = (float *)dst_data;
    int64_t n = 1;                                  /* element count: the product of dst's dims */
    for (int d = 0; d < dst->ndims; d++)
        n *= dst->dims[d];
    for (int64_t i = 0; i < n; i++) {
        float v = x[i];
        y[i] = v < -1.0f ? -1.0f : v > 1.0f ? 1.0f : v;
    }
    return 0;
}
```

What the compiler sees, and what the generated app calls:

```
aten.mm                       → 32×128  (Linear 1, GEMM on the AIE)
custom_op_demo_c::softclip_c  → 32×128  (aiednn_custom_create: your function, on the A72)
aten.mm                       → 32×64   (Linear 2, GEMM on the AIE)
```

```c
st = aiednn_custom_create(&p, 1, c1, &m2, aiednn_custom_custom_op_demo_c_softclip_c_default, nullptr);
```

Every C build of the app, the x86 workstation one as well as the board ELF, compiles your function in. Only the
Python-side reference run (`dnn_torch.run_ctypes`) calls the op through PyTorch itself.

Use this when the op has no torch equivalent (a lookup table, a driver, a codec), or when you want to control
the host implementation exactly (e.g. NEON intrinsics).

---

## Running the example

```bash
# workstation emulation (all three cases)
PYTHONPATH=src python example/aiednn/custom_pytorch_op/run.py

# one case at a time
PYTHONPATH=src python example/aiednn/custom_pytorch_op/run.py --case a    # or b, c

# board run (VEK280)
PYTHONPATH=src python example/aiednn/custom_pytorch_op/run.py \
    --target vek280 --board <host> --vitis-settings <.../Vitis/settings64.sh>
```

## Decision tree

```
Is the op implemented in torch ops?
├── yes → export_inlined()
│         ├── torch.library.custom_op?   →  case A (body found automatically)
│         └── older lib.impl("CPU")?     →  case B (pass bodies=)
└── no  → plain torch.export.export() + custom_sources=[file.c]  →  case C
```

Not sure? Call `export_inlined()` without `bodies=`: if a node with `dnn_export.is_custom_op(node.target)`
is still in `ep.graph`, the op has no Python body torch can find, so use case B or C.
