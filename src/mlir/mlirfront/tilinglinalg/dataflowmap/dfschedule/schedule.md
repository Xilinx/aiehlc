###############################################################################
# Copyright (C) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
/*
three dialect
amd_host (Host: async event, memory, kernel start, H2D/D2H, stream)
amd_dma (Device: DMA/Lock)
flow (orchestrate: kernel, pipeline_sink/source, stream bind)
*/
/*
Pattern 1 — Memory-backed inputs/outputs + scalar parameters
- Inputs/outputs are memrefs in GLOBAL. Host performs H2D/D2H. Kernel pulls/pushes via pipeline sink/source or direct dmasync.

```mlir
// Device kernel: pull input from GLOBAL, compute, write to GLOBAL.
// Also takes scalar params like length and a scale factor.
flow.kernel @kernel_for_tile_B(
  %in_gmem:  memref<4096xf32, "GLOBAL">,
  %out_gmem: memref<4096xf32, "GLOBAL">,
  %length:   index,
  %alpha:    f32
) {
  on_tile = #dma_hop<tile {col=2, row=2}>
} {
  %ping_in  = memref.alloca() : memref<256xf32, "SHARED">
  %pong_in  = memref.alloca() : memref<256xf32, "SHARED">
  %ping_out = memref.alloca() : memref<256xf32, "SHARED">
  %pong_out = memref.alloca() : memref<256xf32, "SHARED">

  // Input side: memory-backed sink (GLOBAL -> SHARED ping-pong)
  flow.pipeline_sink(%in_gmem, %ping_in, %pong_in)
    { io_class = "memory", dma = { direction = "MM2S", channel = 0, burst = 64, align = 64 },
      chunk = 256 }
    yields (%chunk_idx: index, %in_chunk: memref<256xf32, "SHARED">)
  {
    // Compute for this chunk -> write to SHARED out buffer
    "core.compute_scale"(%in_chunk, %ping_out, %alpha)
      : (memref<256xf32, "SHARED">, memref<256xf32, "SHARED">, f32) -> ()

    // Output side: memory-backed source (SHARED -> GLOBAL ping-pong)
    flow.pipeline_source(%out_gmem, %ping_out, %pong_out)
      { io_class = "memory", dma = { direction = "S2MM", channel = 1, burst = 64, align = 64 } }
      uses (%chunk_idx: index, %out_chunk: memref<256xf32, "SHARED">) {
        // The body is optional; source op will DMA %out_chunk to %out_gmem subview for this chunk
      }
  }
}

// Host: allocate, copy H2D, launch with params, copy D2H
func.func @host_main() {
  %in_g  = amd.host.alloc_device_mem : memref<4096xf32, "GLOBAL">
  %out_g = amd.host.alloc_device_mem : memref<4096xf32, "GLOBAL">
  %h_in  = memref.alloc() : memref<4096xf32>
  %h_out = memref.alloc() : memref<4096xf32>

  %e_h2d = amd.host.copy_h2d_async %h_in, %in_g
    { via = #flow.shim_endpoint<tile = #dma_hop<tile {col=2,row=0}>, port="MM2S0", channel=3>,
      dma = { burst = 64, align = 64 } }
    : (memref<4096xf32>, memref<4096xf32, "GLOBAL">) -> !amd.event

  %len   = arith.constant 4096 : index
  %alpha = arith.constant 0.5  : f32

  %e_k = amd.host.launch_kernel_async
           @kernel_for_tile_B(%in_g, %out_g, %len, %alpha) {
           on_tile = #dma_hop<tile {col=2, row=2}>
         } wait(%e_h2d)
         : (memref<4096xf32, "GLOBAL">, memref<4096xf32, "GLOBAL">, index, f32) -> !amd.event

  // Pull results back (D2H); you can pin a specific SHIM S2MM channel
  %e_d2h = amd.host.copy_d2h_async %out_g, %h_out
    { via = #flow.shim_endpoint<tile = #dma_hop<tile {col=2,row=0}>, port="S2MM0", channel=4>,
      dma = { burst = 64, align = 64 } }
    : (memref<4096xf32, "GLOBAL">, memref<4096xf32>) -> !amd.event

  amd.host.wait %e_k  : !amd.event
  amd.host.wait %e_d2h: !amd.event
  return
}
```

When to use:
- Host owns the buffers, you want deterministic H2D/D2H, and you can pin SHIM channels on host ops.

Pattern 2 — Stream-backed I/O + scalar parameters
- Inputs/outputs are streams. Host creates/binds streams (or you bind to device SHIM endpoints) and passes stream handles. Scalars are extra operands.

```mlir
// Device kernel with input and output streams + extra params
flow.kernel @kernel_for_tile_B(
  %in_stream:  !flow.stream<f32>,
  %out_stream: !flow.stream<f32>,
  %length:     index,
  %alpha:      f32
) {
  on_tile = #dma_hop<tile {col=2, row=2}>
} {
  %pin = memref.alloca() : memref<256xf32, "SHARED">
  %pon = memref.alloca() : memref<256xf32, "SHARED">

  // Stream input → ping/pong SHARED
  flow.pipeline_sink(%in_stream, %pin, %pon)
    { io_class = "stream", rate = { tokens_per_chunk = 256, width_bits = 32 } }
    yields (%i: index, %in_chunk: memref<256xf32, "SHARED">)
  {
    // Compute
    %pout0 = memref.alloca() : memref<256xf32, "SHARED">
    %pout1 = memref.alloca() : memref<256xf32, "SHARED">
    "core.compute_scale"(%in_chunk, %pout0, %alpha)
      : (memref<256xf32, "SHARED">, memref<256xf32, "SHARED">, f32) -> ()

    // Emit chunk to output stream
    flow.pipeline_source(%out_stream, %pout0, %pout1)
      { io_class = "stream", rate = { tokens_per_chunk = 256, width_bits = 32 } }
      uses (%i: index, %out_chunk: memref<256xf32, "SHARED">) {
        // The pipeline_source will stream out %out_chunk
      }
  }
}

// Host: create/bind streams (to SHIM endpoints if needed), pass them and scalars
func.func @host_main() {
  %in_s  = amd.host.create_stream { width_bits = 32 } : !flow.stream<f32>
  %out_s = amd.host.create_stream { width_bits = 32 } : !flow.stream<f32>

  // Optionally bind to specific SHIM ports/channels
  flow.stream.bind %in_s
    to #flow.shim_endpoint<tile = #dma_hop<tile {col=2,row=0}>, port="MM2S0", channel=3>
    : !flow.stream<f32>
  flow.stream.bind %out_s
    to #flow.shim_endpoint<tile = #dma_hop<tile {col=2,row=0}>, port="S2MM0", channel=4>
    : !flow.stream<f32>

  %len   = arith.constant 4096 : index
  %alpha = arith.constant 0.5  : f32

  %e_k = amd.host.launch_kernel_async
           @kernel_for_tile_B(%in_s, %out_s, %len, %alpha)
           { on_tile = #dma_hop<tile {col=2, row=2}> }
           : (!flow.stream<f32>, !flow.stream<f32>, index, f32) -> !amd.event

  // Host produces input stream and consumes output stream
  %hin  = memref.alloc() : memref<4096xf32>
  %hout = memref.alloc() : memref<4096xf32>
  amd.host.stream_push %in_s, %hin  : (!flow.stream<f32>, memref<4096xf32>) -> ()
  amd.host.stream_pull %out_s, %hout : (!flow.stream<f32>, memref<4096xf32>) -> ()

  amd.host.wait %e_k : !amd.event
  return
}
```

When to use:
- You need live streaming semantics (overlap host/device, minimal buffering).
- You must pin SHIM endpoints for streams rather than buffer DMA.

Pattern 3 — Mixed I/O (e.g., stream input, memory-backed output) + extra parameters
- Combine as needed; streams for ingress and GLOBAL for egress (or vice versa).

```mlir
flow.kernel @kernel_for_tile_B(
  %in_stream: !flow.stream<f32>,
  %out_gmem:  memref<4096xf32, "GLOBAL">,
  %alpha:     f32
) {
  on_tile = #dma_hop<tile {col=2, row=2}>
} {
  %pin  = memref.alloca() : memref<256xf32, "SHARED">
  %pon  = memref.alloca() : memref<256xf32, "SHARED">
  %pout = memref.alloca() : memref<256xf32, "SHARED">
  %qout = memref.alloca() : memref<256xf32, "SHARED">

  flow.pipeline_sink(%in_stream, %pin, %pon)
    { io_class = "stream", rate = { tokens_per_chunk = 256, width_bits = 32 } }
    yields (%i: index, %in_chunk: memref<256xf32, "SHARED">)
  {
    "core.compute_scale"(%in_chunk, %pout, %alpha)
      : (memref<256xf32, "SHARED">, memref<256xf32, "SHARED">, f32) -> ()

    flow.pipeline_source(%out_gmem, %pout, %qout)
      { io_class = "memory",
        dma = { direction = "S2MM", channel = 1, burst = 64, align = 64 } }
      uses (%i: index, %out_chunk: memref<256xf32, "SHARED">) { }
  }
}
```

 need to handle anything special?
- Kernel signature: Yes, declare all needed operands explicitly (memrefs, streams, scalars). You do not need to manually implement ping-pong or DMA—flow.pipeline_* plus your lowering passes should handle that.
- Host side: Yes, you must provide actual operands (allocate/bind/copy and pass them at launch). For streams, create/bind stream handles; for memrefs, allocate/copy; for scalars, pass values.
- Attributes: If you must pin transport details (SHIM tile, port, channel), put them on:
  - host copy ops (for memory-backed H2D/D2H), or
  - stream binding ops (for stream-backed I/O).
  Keep kernel portable when possible; avoid baking SHIM details into the kernel unless required.

Lowering guidelines
- flow.kernel → func.func on device with the same operands (memref spaces preserved, stream handles lowered to device stream endpoints).
- flow.pipeline_sink/source:
  - memory-backed: lower to dma_start (+ optional 2D) + lock.acquire + ping-pong management.
  - stream-backed: lower to stream endpoint config + switch routing; SHARED ping-pong buffers for bursty compute.
- Scalars: pass-through operands; use directly in compute; no special handling.

Verification best practices
- Mark memref operand intent if helpful: read_only/write_only attributes (or memory effects interfaces) so the compiler can validate no unintended aliasing.
- Ensure chunk size divides length or add tail handling (compiler pass can insert a remainder loop).
- Define burst/alignment consistent with platform recommendation (e.g., 64B).

In short: yes, declare inputs, outputs, and other parameters on low.kernel (or flow.kernel). Handle host-side provisioning and, where needed, binding/pinning of transport. Let the lowering pass handle DMA/locks/streams around your pipeline ops.

Follow-up questions you could ask:
- Do you want a pass skeleton that rewrites flow.pipeline_{sink,source} into L1 dma + lock ops?
- Should I add tail-handling (non-multiple-of-chunk) to the examples?
- Do you want examples with 2D tiles (width/height/strides) and mixed stream/memory I/O pinned to specific SHIM channels?
*/
/*
func.func @kernel_tile_B_basic(%in_gmem: memref<1024xf32, "GLOBAL">) {

  %smem = memref.alloca() : memref<1024xf32, "SHARED">

  %lk = amd.dma.lock.init(0 : i32) : !amd.dma.lock

  amd.dma.dma_start %in_gmem, %smem,
    direction = #amd.dma<MM2S>, channel = 0, priority = 0, burst = 64,
    release_lock_on_done = %lk, release_value = 1 : i32
    : (memref<1024xf32, "GLOBAL">, memref<1024xf32, "SHARED">) -> ()


  amd.dma.lock.acquire %lk, 1 : (!amd.dma.lock, i32) -> ()

  %c0 = arith.constant 0 : index
  %cN = arith.constant 100 : index
  %c1 = arith.constant 1 : index
  scf.for %i = %c0 to %cN step %c1 {
    "core.compute"(%smem) : (memref<1024xf32, "SHARED">) -> ()
  }
  return
}

func.func @kernel_tile_B_pingpong(%in_gmem: memref<1024xf32, "GLOBAL">) {
  %ping = memref.alloca() : memref<256xf32, "SHARED">
  %pong = memref.alloca() : memref<256xf32, "SHARED">
  %lk   = amd.dma.lock.init(0 : i32) : !amd.dma.lock

  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c256 = arith.constant 256 : index

  %sv0 = memref.subview %in_gmem[%c0] [256] [1]
         : memref<1024xf32, "GLOBAL"> to memref<256xf32, "GLOBAL">
  amd.dma.dma_start %sv0, %ping,
    direction = #amd.dma<MM2S>, channel = 0, burst = 64,
    release_lock_on_done = %lk, release_value = 1 : i32
    : (memref<256xf32, "GLOBAL">, memref<256xf32, "SHARED">) -> ()

  scf.for %t = %c0 to %c4 step %c1 {
    %need = arith.addi %t, %c1 : index
    %need_i32 = arith.index_cast %need : index to i32
    amd.dma.lock.acquire %lk, %need_i32 : (!amd.dma.lock, i32) -> ()

    %isOdd = arith.remsi %t, arith.constant 2 : index
    scf.if (arith.cmpi(eq, %isOdd, %c0)) {
      "core.compute"(%ping) : (memref<256xf32, "SHARED">) -> ()
    } else {
      "core.compute"(%pong) : (memref<256xf32, "SHARED">) -> ()
    }

  
    %last = arith.cmpi(eq, %t, arith.constant 3 : index)
    scf.if (arith.cmpi(ne, %last, arith.constant true)) {
      %t1 = arith.addi %t, %c1 : index
      %off = arith.muli %t1, %c256 : index
      %svN = memref.subview %in_gmem[%off] [256] [1]
             : memref<1024xf32, "GLOBAL"> to memref<256xf32, "GLOBAL">
      %rel = arith.addi %need_i32, arith.constant 1 : i32
      %t1i = arith.index_cast %t1 : index to i32
      %usePong = arith.cmpi(ne,
                      arith.remsi %t1i, arith.constant 2 : i32),
                      arith.constant 0 : i32)

      scf.if %usePong {
        amd.dma.dma_start %svN, %pong,
          direction = #amd.dma<MM2S>, channel = 0, burst = 64,
          release_lock_on_done = %lk, release_value = %rel : i32
          : (memref<256xf32, "GLOBAL">, memref<256xf32, "SHARED">) -> ()
      } else {
        amd.dma.dma_start %svN, %ping,
          direction = #amd.dma<MM2S>, channel = 0, burst = 64,
          release_lock_on_done = %lk, release_value = %rel : i32
          : (memref<256xf32, "GLOBAL">, memref<256xf32, "SHARED">) -> ()
      }
    }
  }
  return
}
*/
//Version 3
/*
// "After" Pass 2 (SchedulePass)
builtin.module {
  
  // (the "contract" and "script" generated by Pass 1 still exist)
  routinghw.config @my_routes {
    routinghw.packet_flow @route_A {
       dest_tile   = #dma_hop<tile {col=0, row=3}>,
       packet_id   = 1 : i32
    }
    routinghw.packet_flow @route_B {
       dest_tile   = #dma_hop<tile {col=1, row=3}>,
       packet_id   = 2 : i32
    }
  }
  func.func @main() { ... } // (routing hardware configuration script)

  // --- **Output of Pass 2 (SchedulePass) (host-side code)** ---
  //
  // this func.func is *newly generated* by Pass 2
  func.func @host_main_schedule() {
    %gmem = "ds.host.alloc_device_mem"() : memref<1024xf32, 1>

    // (Pass 2 get the handle from dmaphop)
    %shim_handle = "ds.host.get_tile_handle"() { col = 2, row = 0 }
    %core_A_handle = "ds.host.get_tile_handle"() { col = 0, row = 3 }
    %core_B_handle = "ds.host.get_tile_handle"() { col = 1, row = 3 }

    // --- This is where @route_A is used ---
    // 1. "Schedule" Pass generates code, 
    //    which *looks up* the route from the "routinghw" (Pass 1) contract.
    //    This operation converts a *compile-time symbol* (@route_A)
    //    into a *runtime handle* (%stream_A).
    %stream_A = "ds.host.get_stream_handle"(@my_routes::@route_A)
      : () -> !ds.stream
    %stream_B = "ds.host.get_stream_handle"(@my_routes::@route_B)
      : () -> !ds.stream

    // 2. "Schedule" Pass generates code to *set up DMA*.
    //    This "launch_dma" operation *uses* the runtime handles
    //    (%stream_A and %stream_B) verified by Pass 1.
    %evt_dma = "ds.host.launch_dma_g2s_async" on %shim_handle (
        %gmem, %stream_A, %stream_B  // <-- pass the handles
      ) : (memref<...>, !ds.stream, !ds.stream) -> !ds.event

    // 3. "Schedule" Pass generates code to *launch the kernel*.
    //    (The kernel is also passed the handle it needs to listen on)

    %pid_A = "routinghw.get_packet_id"(@my_routes::@route_A) : i32
    %pid_B = "routinghw.get_packet_id"(@my_routes::@route_B) : i32
    
    %evt_A = "ds.host.launch_kernel_async" on %core_A_handle (
        @dskernel_A, %pid_A // <-- pass the packet ID
      ) : (i32) -> !ds.event
      
    %evt_B = "ds.host.launch_kernel_async" on %core_B_handle (
        @dskernel_B, %pid_B
      ) : (i32) -> !ds.event

    "ds.host.wait"(%evt_dma, %evt_A, %evt_B)
    return
  }
  
  // (Pass 2 also generates the L1 dskernel ...)
}
*/
//version 3 , hop through memtile
/*
// --- The Scheduler (Generated by Pass 2) ---
  func.func @host_main() {
    %gmem = "ds.host.alloc_device_mem"() : ...

    // 1. Get handles for all 3 physical tiles involved
    %shim_handle = "ds.host.get_tile_handle"() { col = 2, row = 0 }
    %memtile_handle = "ds.host.get_tile_handle"() { col = 4, row = 0 }
    %coretile_handle = "ds.host.get_tile_handle"() { col = 4, row = 2 }

    // 2. Read the "contract" to get the DMA config (packet IDs)
    %pid_hop1 = "routinghw.get_packet_id"(@my_cascaded_routes::@route_hop1) : i32
    %pid_hop2 = "routinghw.get_packet_id"(@my_cascaded_routes::@route_hop2) : i32

    // --- 3. Launch the 3 parallel tasks ---
    
    // Task A: Launch Gmem-DMA (DDR -> Mem Tile)
    // "Push %gmem from Shim(2,0) tagged with ID 10"
    %evt_dma = "ds.host.launch_dma_g2s_async" on %shim_handle (
        %gmem, %pid_hop1
      ) : (memref<...>, i32) -> !ds_host.event

    // Task B: Launch the Mem Tile "Passthrough" Kernel
    // "Tell MemTile(4,0) to start,
    //  listen for ID 10, and send on ID 11"
    %evt_memtile = "ds.host.launch_kernel_async" on %memtile_handle (
        @dskernel_memtile_passthrough, %pid_hop1, %pid_hop2
      ) : (i32, i32) -> !ds_host.event

    // Task C: Launch the Core Tile "Compute" Kernel
    // "Tell CoreTile(4,2) to start,
    //  and listen for ID 11"
    %evt_coretile = "ds.host.launch_kernel_async" on %coretile_handle (
        @dskernel_coretile_compute, %pid_hop1
      ) : (i32) -> !ds_host.event
        
    // 4. Synchronize: Wait for all 3 tasks to finish
    "ds.host.wait"(%evt_dma, %evt_memtile, %evt_coretile)
    
    "ds.host.free_device_mem"(%gmem)
    return
  }
*/
/*
// --- Kernel 1: The "Passthrough" Kernel (Runs on Mem Tile) ---
  // This kernel *is* the L2 decoupling buffer.
  // It pipelines an S2M DMA with an M2S DMA.
  func.func @dskernel_memtile_passthrough(
      %pktid_in: i32,  // Receives 10
      %pktid_out: i32 // Receives 11
    ) {
    // This tile's "decoupling buffer" (e.g., 256k smem)
    // is split into Ping-Pong
    %ping = memref.alloca() : memref<128kxf32, "SHARED">
    %pong = memref.alloca() : memref<128kxf32, "SHARED">

    // (Initialize 4 locks: S2M_Ready, S2M_Done, M2S_Ready, M2S_Done)
    %lock_s2m_rdy = "dskernel.lock_init"(1) ...
    %lock_s2m_done = "dskernel.lock_init"(0) ...
    %lock_m2s_rdy = "dskernel.lock_init"(0) ...
    %lock_m2s_done = "dskernel.lock_init"(1) ...

    // 1. Launch S2M DMA (Hop 1: Stream -> Smem)
    //    Listens on ID 10
    "dskernel.launch_dma_s2m_loop"(%ping, %pong) {
      listen_on_packet_id = %pktid_in,
      dma_wait_lock = %lock_s2m_rdy,
      dma_release_lock = %lock_s2m_done
    } ...
    
    // 2. Launch M2S DMA (Hop 2: Smem -> Stream)
    //    Sends on ID 11
    "dskernel.launch_dma_m2s_loop"(%ping, %pong) {
      send_as_packet_id = %pktid_out,
      dma_wait_lock = %lock_m2s_rdy,
      dma_release_lock = %lock_m2s_done
    } ...
    
    // 3. "Compute" loop (just flips locks)
    //    This loop simply manages the passthrough
    scf.for %i = 0 to ... {
      // Wait for S2M to fill a buffer
      "dskernel.acquire_lock"(%lock_s2m_done, 1)
      // Wait for M2S to be ready for a buffer
      "dskernel.acquire_lock"(%lock_m2s_done, 1)

      // (If we were doing data re-org, it would happen here)

      // Tell M2S to send the buffer
      "dskernel.release_lock"(%lock_m2s_rdy, 1)
      // Tell S2M it can re-fill the (now empty) buffer
      "dskernel.release_lock"(%lock_s2m_rdy, 1)
    }
    return
  }
  
  // --- Kernel 2: The "Compute" Kernel (Runs on Core Tile) ---
  // This is the final consumer
  func.func @dskernel_coretile_compute(
      %pktid_in: i32 // Receives 11
    ) {
    // This tile's L1 Ping-Pong buffer (e.g., 16k smem)
    %ping = memref.alloca() : memref<8kxf32, "SHARED">
    %pong = memref.alloca() : memref<8kxf32, "SHARED">

    // (Initialize locks for S2M <-> Compute)
    %lock_dma = "dskernel.lock_init"(1) ...
    %lock_compute = "dskernel.lock_init"(0) ...

    // 1. Launch local S2M DMA (Hop 2: Stream -> Smem)
    //    Listens on ID 11
    "dskernel.launch_dma_s2m_loop"(%ping, %pong) {
      listen_on_packet_id = %pktid_in,
      dma_wait_lock = %lock_dma,
      dma_release_lock = %lock_compute
    } ...

    // 2. Compute loop
    scf.for %i = 0 to ... {
      // Wait for local DMA to fill a buffer
      "dskernel.acquire_lock"(%lock_compute, 1)

      // Compute (Smem reuse)
      scf.for %j = 0 to 10 { "core.compute"(...) }

      // Release buffer for local DMA to refill
      "dskernel.release_lock"(%lock_dma, 1)
    }
    return
  }
}
*/
