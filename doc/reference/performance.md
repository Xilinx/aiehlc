# Performance: control-plane cost

(Moved from the root README.md.)

## Performance data

The cost of a launch comes from three domains: compute, data movement and
control. The control cost is register reads / writes plus the host's own
algorithm cost on the CPU; the latter is low, so most of it is the register
writes, which by default go through AXI-MM one at a time. The documents below
measure the control plane.

[Register access unit cost — ~372 ns/write AXI-MM vs ~2.9 ns/write control packet](../performance/register_write_cost.md)

[Register r/w perf in axi-mm and control pkt](../performance/controlperf_analysis.md)

[Control-plane timeline](../performance/controlperf_timeline_crosstrack.md)


```
==== AIE control-plane API microbenchmark (AIE_GEN=5) ====
Each __Runtime_* config call = a burst of AXI-MM register writes to the device.
API                                   iters     total_us      us/call
-------------------------------------------------------------------------
XAie_TileLoc (cpu)                     1000         3.19       0.0032
XAie_DmaDescInit (cpu)                 1000        19.09       0.0191
dma_createio_4 (cpu struct)            1000        39.90       0.0399
-------------------------------------------------------------------------
API vs raw write32              nW     api_us     raw_us   delta_us     ns/wr
-------------------------------------------------------------------------
dma_bd_config (shim,1024)        9     3.3388     3.3522    -0.0134     372.5
dma_bd_config (core,1024)        6     2.2914     2.3012    -0.0098     383.5
dma_bd_config_multidim (3D)      9     3.3306     3.3606    -0.0300     373.4
dma_bd_config_multidim_ooo       9     3.3394     3.3517    -0.0123     372.4
XAie_DmaWriteBd (shim)           9     3.3366     3.3536    -0.0170     372.6
XAie_LockSetValue+LockInit       1     0.3712     0.3838    -0.0126     383.8
dma_channel_enable_ooo           1     0.3622     0.3714    -0.0092     371.4
startio (SetStartQueue)          1     0.3645     0.3684    -0.0039     368.4
wait_io (idle poll, read)        1     0.8796     0.4143     0.4653     414.3
load_kernel_group_16t (ELF x16)          20    147415.26    7370.7631
launch_kernel_group (enable x16)         20       276.15      13.8076
-------------------------------------------------------------------------
--- control overhead / matmul iter (120 bd_config + 60 createio+startio) ---
      api=290.11 us   raw(780 write32)=296.04 us   host_overhead=-5.93 us   raw=379.5 ns/write
==== control-plane microbenchmark done ====
```

## Control packets vs raw AXI-MM writes

Everything below is measured on real hardware, not estimated. If you need to
justify a control-plane design decision, cite this page rather than re-deriving
a number from a timeline span (see [Pitfalls](../performance/register_write_cost.md#pitfalls)).

| | Path | ns / 32-bit write |
|---|---|---|
| Today | host `XAie_Write32` (serial MMIO) | **~372** |
| Batched | control packet via shim DMA | **~2.9** |
