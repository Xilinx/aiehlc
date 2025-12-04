/******************************************************************************
* Copyright (C) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
* SPDX-License-Identifier: MIT
******************************************************************************/
Blueprint
module {
  dfscheblueprint.config @broadcast_blueprint {

    // ==========================================
    // 1. Physical Resource Definition
    // ==========================================
    dfscheblueprint.tile_group @shim_tx { tiles = [(0, 2)] }
    dfscheblueprint.tile_group @core_rx { tiles = [(2, 0), (2, 1)] }

    // ==========================================
    // 2. Logical Data Definition (Using Tensor)
    // ==========================================
    // Declare total data (1024 float)
    %data = dfscheblueprint.declare_data <1024xf32>

    // Define partition view (Tile Size = 512)
    // Slice 0: [0, 512], Slice 1: [512, 512]
    %view = dfscheblueprint.partition %data { tile_shape = [512] }

    // ==========================================
    // 3. Binding
    // ==========================================
    
    // [Bind A]: Shim Sender (Holds Root Data)
    dfscheblueprint.flowconfig @bind_shim {
      target = @shim_tx,
      view   = %view,
      distribution = "root",
      dma    = #dfscheblueprint.DMA<channel=0, direction=MM2S>
    }

    // [Bind B]: Core Receiver (Holds Sliced Data)
    // Semantics: Core 0 and Core 1 both receive *complete* Root Data (Broadcast)
    //      (If Scatter, distribution="linear" would be used here)
    dfscheblueprint.flowconfig @bind_cores {
      target_group = @core_rx,
      view         = %view,
      distribution = "replicate", // Broadcast mode
      dma          = #dfscheblueprint.DMA<channel=0, direction=S2MM>
    }

    // ==========================================
    // 4. Transfer
    // ==========================================
    // Circuit-switched broadcast (No Packet ID)
    dfscheblueprint.flow_transfer @op_broadcast {
      type = "one_to_many",
      from = @bind_shim,
      to   = @bind_cores,
      routing_mode = "circuit"
    }
  }


  ----------after transfer-----
  // Host Main Program
  func.func @host_main() {
    
    // ============================================================
    // 1. HOST CONFIG SECTION (One-time setup)
    // ============================================================
    // Returns Context handles (Including Shim, Cores, Memory)
    %ctx_shim, %ctx_c0, %ctx_c1 = df.host.config.section() -> (!df.tile, !df.tile, !df.tile) {
      
      // A. Define Physical Tile Symbols
      df.host.config.define_tile @shim_02 {col=0, row=2}
      df.host.config.define_tile @core_20 {col=2, row=0}
      df.host.config.define_tile @core_21 {col=2, row=1}

      // B. Allocate DDR Memory (Corresponds to Blueprint %data)
      df.host.config.alloc_buffer @gmem_data {size=1024, type=f32}

      // C. Configure Shim DMA (MM2S)
      //    Derived from Blueprint: Channel 0, Circuit Mode
      df.host.config.dma_bd @shim_02 {
         channel = 0, bd_id = 0,
         buffer = @gmem_data, offset = 0, len = 1024,
         enable_packet = false, // Circuit Mode
         next_bd = -1
      }

      // D. Resolve symbols to handles and Yield
      %s = df.host.config.lookup_tile @shim_02
      %c0 = df.host.config.lookup_tile @core_20
      %c1 = df.host.config.lookup_tile @core_21
      
      df.yield %s, %c0, %c1 : !df.tile, !df.tile, !df.tile
    }

    // ============================================================
    // 2. HOST SCHEDULE SECTION (Runtime Loop)
    // ============================================================
    // Pass handles generated in Config phase
    scf.for %i = 0 to 100 {
      df.host.dfscheblueprint.section(%ctx_shim, %ctx_c0, %ctx_c1) {
        ^bb0(%shim: !df.tile, %c0: !df.tile, %c1: !df.tile):

        // A. Start Shim Send (Kick-off)
        df.host.dfscheblueprint.dma_push %shim {
           channel=0, direction="MM2S", bd_id=0
        }

        // B. Start Core 0 Receive
        %e0 = df.host.dfscheblueprint.kernel_launch %c0 { 
           callee=@dskernel_recv 
        } : (!df.tile) -> !df.event

        // C. Start Core 1 Receive
        %e1 = df.host.dfscheblueprint.kernel_launch %c1 { 
           callee=@dskernel_recv 
        } : (!df.tile) -> !df.event

        // D. Wait
        df.host.dfscheblueprint.wait(%e0, %e1)
        
        df.yield
      }
    }
    return
  }

  ----kernel
  // Kernel Function (Runs on Tile)
  func.func @dskernel_recv() {

    // ============================================================
    // 1. KERNEL CONFIG SECTION (Initialization)
    // ============================================================
    // Yield Buffer and Lock for later use
    %ping, %pong, %acq, %rel = df.kernel.config.section() -> (
        memref<512xf32, "SHARED">, memref<512xf32, "SHARED">, !df.lock, !df.lock
    ) {
      // A. Allocate local Ping-Pong (Size 512, from Blueprint Tile Size)
      %p1 = memref.alloca() : memref<512xf32, "SHARED">
      %p2 = memref.alloca() : memref<512xf32, "SHARED">

      // B. Initialize Locks (S2MM Mode)
      //    Lock 0 (Acquire): 0=Empty(DMA writable), 1=Full(Not writable)
      //    Lock 1 (Release): 0=Empty(Core not readable), 1=Full(Core readable)
      %la = df.kernel.config.lock_init {id=0, val=0}
      %lr = df.kernel.config.lock_init {id=1, val=0}

      // C. Configure local BD Chain (Ping <-> Pong)
      //    BD 0 (Ping)
      df.kernel.config.dma_bd {
         bd_id=0, buffer=%p1, len=512,
         locks={acq=%la, acq_val=0, rel=%lr, rel_val=1}, // Wait 0, Set 1
         next_bd=1
      }
      //    BD 1 (Pong)
      df.kernel.config.dma_bd {
         bd_id=1, buffer=%p2, len=512,
         locks={acq=%la, acq_val=0, rel=%lr, rel_val=1},
         next_bd=0
      }

      df.yield %p1, %p2, %la, %lr : memref<512xf32, "SHARED">, memref<512xf32, "SHARED">, !df.lock, !df.lock
    }

    // ============================================================
    // 2. KERNEL SCHEDULE SECTION (Runtime)
    // ============================================================
    df.kernel.dfscheblueprint.section(%ping, %pong, %acq, %rel) {
      ^bb0(%b_ping: memref<512xf32, "SHARED">, 
           %b_pong: memref<512xf32, "SHARED">, 
           %l_acq: !df.lock, 
           %l_rel: !df.lock):

      // A. Start local DMA Engine (S2MM Channel 0)
      df.kernel.dfscheblueprint.dma_push {
         channel=0, direction="S2MM", start_bd_id=0
      }

      // B. Compute Loop (Ping-Pong Processing)
      //    Assume processing 10 rounds of data
      scf.for %iter = 0 to 10 {
         
         // 1. Select current Buffer (Ping or Pong)
         %is_even = arith.remsi %iter, 2 : index
         %curr_buf = scf.if %is_even -> (memref<512xf32, "SHARED">) {
            scf.yield %b_ping : memref<512xf32, "SHARED">
         } else {
            scf.yield %b_pong : memref<512xf32, "SHARED">
         }

         // 2. [Sync] Wait for DMA to fill data (Wait for Lock == 1)
         df.kernel.dfscheblueprint.lock_acquire %l_rel, 1

         // 3. [Compute] Compute
         "core.compute"(%curr_buf) : (memref<512xf32, "SHARED">) -> ()

         // 4. [Sync] Return Buffer to DMA (Set Lock = 0)
         df.kernel.dfscheblueprint.lock_release %l_acq, 0
      }
      
      df.yield
    }
    return
  }
}
