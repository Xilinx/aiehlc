# Control-Plane Pragmas

(Moved from the root README.md.)

Two opt-in marker pragmas (no arguments) enable control-plane features in the
multi-tile (`tilinglinalg`) flow. Both are off by default; add the pragma to your
source file to turn the feature on. They are detected during preprocessing and
published as module attributes that gate the corresponding MLIR pipeline stage.

```cpp
#pragma control_plan_op_control_packet
#pragma CONTROL_PLAN_GROUP_REG_WRITE

__global__ void mykernel(const int32_t *A, int32_t *B) {
    // kernel code
}

int main() {
    // host code
}
```

| Pragma | Module attr | Effect when present |
|--------|-------------|---------------------|
| `#pragma control_plan_op_control_packet` | `routing.control_plan_op_control_packet` | Reserves control-plane stream-switch resources and lowers kernel ELF loading plus core launch to fire-and-forget broadcast control-packet writes. Absent => normal `XAie_LoadElfMem` / `XAie_CoreEnable` behavior. |
| `#pragma CONTROL_PLAN_GROUP_REG_WRITE` | `routing.control_plan_group_reg_write` | The host pipeline runs `GroupRegWritePass`, which coalesces identical core-tile lock-init register writes into **control-packet group writes** (broadcast / row-multicast, lowered to `__Runtime_ctrl_row_write_ack`). Absent => the pass is skipped and lock inits are emitted as individual `XAie_LockSetValue` register writes. |

## `#pragma control_plan_op_control_packet`

Reserves control-plane resources so a control-packet fabric (see the row-control
planner APIs in `aie_runtime.h`) can share the array with the data plane without
resource conflicts. The pipeline gates `reserveControlPlaneResources(...)` on the
published attr.

The host pipeline also materializes the row-control fabric before
`load_kernel_group`. Generated code calls
`__Runtime_load_kernel_group_{4,8,16}t_ctrl` to stream ELF program/data words
through the reserved fabric, then calls `__Runtime_launch_kernel_group_ctrl` to
enable the cores through that same path. Core reset/unreset remains direct
driver configuration so the CTRL endpoint is live while the ELF is transferred.
ELF PT_LOAD bytes are pktized as 4-word broadcast writes. On AIE2PS,
`__Runtime_ctrl_high_throughput_enable(1)` before device initialization disables
CTRL TLAST errors at partition init and packs all self-delimiting accesses into
one shim MM2S BD with one final TLAST. The last write-with-return stays separate
and its CTRL ACK drain is the completion barrier. Without that opt-in, the
compatibility path uses one BD and TLAST per access. Generated control-plane
traffic uses the hardware-validated shim MM2S/S2MM channel 0 pair used by
`ctrlrow_demo`.

## `#pragma CONTROL_PLAN_GROUP_REG_WRITE`

Enables `GroupRegWritePass`. In a tiled GEMM the per-tile lock inits are identical
across the mesh; when this pass runs it clusters them by `(tile-local addr, value)`
and folds a cluster that covers all configured tiles into a single `broadcast`
group write, or all columns of one row into a `row` group write. Each group write
lowers to a blocking `__Runtime_ctrl_row_write_ack` call in `host.cc`. Without the
pragma the pass does not run (the pipeline logs
`GroupRegWritePass skipped (enable with #pragma CONTROL_PLAN_GROUP_REG_WRITE)`)
and every lock init is emitted individually.

Both pragmas work in the single-kernel and multi-kernel `tilinglinalg` paths.

Compare MMIO vs control-packet host setup with the same 4×4 GEMM kernel:

```bash
source script/aiehlc.sh --platform baremetal --aie-version 5 --profiling --skip-bss \
    --runtime-source-file ./example/tileprogram/ccode/simplematmul2.cc
# copy aout/main.elf, then rebuild with simplematmul_ctrl_pkt.cc
```

`simplematmul2.cc` keeps per-tile `XAie_LoadElfMem` / `XAie_LockSetValue` /
`XAie_CoreEnable`. `simplematmul_ctrl_pkt.cc` enables both pragmas so lock
init, ELF load, and core launch use the row-control fabric. Both print
`aie matmul time` plus a `[PERF]` kload/elf/rst/bdcfg/coreen/startio/wait_io
line when `--profiling` is set.
