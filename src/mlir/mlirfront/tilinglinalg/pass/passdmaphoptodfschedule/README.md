# Pass: DmaphopTodfschedulePass

## Overview
This pass converts `routing.RoutingCreate` operations containing `dmaphop` dialect operations into `dfschedule` dialect operations for AIE dataflow scheduling.

## Location
- Header: `src/mlir/mlirfront/tilinglinalg/pass/passdmaphoptodfschedule/passdmaphoptodfschedule.h`
- Implementation: `src/mlir/mlirfront/tilinglinalg/pass/passdmaphoptodfschedule/passdmaphoptodfschedule.cpp`

## Purpose
Transforms physical dataflow operations (dmaphop) into kernel-side scheduling operations (dfschedule) that implement ping-pong buffering for efficient data streaming on AIE tiles.

## Input IR Structure
```mlir
routing.RoutingCreate<Memo = "row"> (scf_idx = %arg : i32) -> i32 {
  ^bb0(%arg0: i32):
    // dmaphop operations: tiles, ports, paths, buffers, pull operations
    routing.yield
}
```

## Output IR Structure
```mlir
module {
  func.func @dskernel_coretile_compute(%arg0: i32) {
    // Ping-pong buffers
    %ping = memref.alloca() {buffer_type = "ping"} : memref<256xf32>
    %pong = memref.alloca() {buffer_type = "pong"} : memref<256xf32>
    
    // Lock initialization
    %ping_acquire = dfschedule.lock_init(0, "ping_aquire_lock") -> !dfschedule.lock
    %pong_acquire = dfschedule.lock_init(0, "pong_aquire_lock") -> !dfschedule.lock
    %ping_release = dfschedule.lock_init(1, "ping_release_lock") -> !dfschedule.lock
    %pong_release = dfschedule.lock_init(0, "pong_release_lock") -> !dfschedule.lock
    
    // DMA thread (runs in parallel)
    dfschedule.launch_dma_s2m_loop %ping, %pong, %arg0, 
                                   %ping_acquire, %pong_acquire, 
                                   %ping_release, %pong_release : {...}
    
    // Compute thread with ping-pong logic
    scf.for %i = %c0 to %c4 step %c1 {
      // Dynamic buffer/lock selection
      %is_even = arith.cmpi "eq", arith.remui(%i, %c2), %c0
      %buffer = scf.if %is_even -> (memref<256xf32>) { ... }
      %acquire_lock = scf.if %is_even -> (!dfschedule.lock) { ... }
      %release_lock = scf.if %is_even -> (!dfschedule.lock) { ... }
      
      dfschedule.acquire_lock %acquire_lock, %lock_val
      scf.for %j = %c0 to %c10 step %c1 {
        dfschedule.compute %buffer : memref<256xf32>
      }
      dfschedule.release_lock %release_lock, %lock_val
    }
    return
  }
}
```

## Key Transformations
1. **Extract dmaphop operations**: Identify tiles, ports, paths, and data movement operations
2. **Create dfschedule module**: Generate module wrapper with dskernel function
3. **Generate ping-pong buffers**: Create dual buffers for double-buffering pattern
4. **Initialize locks**: Set up acquire/release locks for synchronization
5. **Launch DMA thread**: Create parallel DMA loop for data streaming
6. **Implement compute logic**: Generate compute loop with dynamic buffer selection

## Integration
To use this pass in the build system, add to `src/mlir/mlirfront/CMakeLists.txt`:

```cmake
list(APPEND SOURCE_LIB_FILES ./tilinglinalg/pass/passdmaphoptodfschedule/passdmaphoptodfschedule.cpp)
```

And include necessary dialect directories:
```cmake
include_directories(./tilinglinalg/dataflowmap/dfschedule/inc)
include_directories(./tilinglinalg/dataflowmap/dfschedule/)
```

## Pass Registration
```cpp
StringRef getArgument() const final { return "lower-dmaphop-to-dfschedule"; }
StringRef getDescription() const final { return "Lower dmaphop dialect to dfschedule dialect"; }
```

## Dependencies
- `dmaphop` dialect (input)
- `dfschedule` dialect (output)
- `routing` dialect (for RoutingCreate op)
- Standard dialects: `func`, `memref`, `arith`, `scf`

## Usage Example
```bash
# Apply pass during compilation pipeline
mlir-opt --lower-dmaphop-to-dfschedule input.mlir -o output.mlir
```

## Notes
- Lock values increment with loop iterations (1, 2, 3, 4 for iterations 0, 1, 2, 3)
- Ping-pong selection based on even/odd iteration parity
- DMA and compute threads run in parallel using separate lock synchronization
- Buffer size defaults to 256xf32 (configurable if needed)
