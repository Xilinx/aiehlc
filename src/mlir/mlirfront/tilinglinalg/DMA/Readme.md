###############################################################################
# Copyright (C) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: MIT
###############################################################################

# Architecture

top-down compiler architecture we have designed, presented entirely in English.

### **The Complete Compilation Flow (Top-Down Architecture Diagram)**

The following text-based illustration shows the entire compilation flow, from a high-level `linalg` operation down to the final build artifacts.

```

+---------------------------------------------------------------------------------------------------------+
|                           **Stage 1: Unified Intermediate Representation (UIR)** |
|---------------------------------------------------------------------------------------------------------|
| // Physical topology, logical dataflow, and compute co-exist, linked by SSA values.                      |
| scf.for %i = ... {                                                                                      |
|   // Physical Topology Aspect (routinghw)                                                               |
|   %compute_tile = routinghw.tilecreate {row=%i, ...} -> !routinghw.tile                                  |
|                                                                                                         |
|   // Logical Dataflow Aspect (dmap)                                                                     |
|   %tile_port = dmap.configure_port on %compute_tile ... -> !dmap.port                                    |
|   %stream = dmap.create_stream %ddr_port, %tile_port -> !dmap.stream                                     |
|   dmap.push %ddr_slice to %stream -> (into %local_mem)                                                  |
.                                                                |
| }                                                                                                       |
+---------------------------------------------------------------------------------------------------------+
                                                     |
                                                     | Compilation flow "forks" here
                       /-----------------------------------------------------------\
                      /                                                             \
                     v                                                               v
+------------------------------------------+                  +--------------------------------------------------------+
|   **Path A: Generate Static Routing Config** |                  |        **Path B: Generate Dynamic Host Code** |
+------------------------------------------+                  +--------------------------------------------------------+
                     |                                                               |
                     |                                                               | Pass: -lower-dmap-to-hops
                     v                                                               v
+------------------------------------------+                  +--------------------------------------------------------+
|  **Stage A.1: Pure Physical Connection IR** |                  | **Stage B.1: Physical Path Planning Layer (dma_hop)** |
|------------------------------------------|                  |--------------------------------------------------------|
| // Only physical connection ops from     |                  | // dmap is replaced by dma_hop, introducing MemTile     |
|                                          |                  | // caching.                                            |
|                                          |                  | scf.for %i = ... {                                     |
|                                          |                  |   %memtile_buf = dma_hop.alloc_buffer ...              |
|                                          |                  |   dma_hop.transfer %ddr_slice, %memtile_buf            |
| }                                        |                  |   dma_hop.wait ...                                     |
+------------------------------------------+                  |   dma_hop.transfer %memtile_buf, %local_mem            |
                     |                                        |   dma_hop.wait ...                                     |
                     | Pass: -lower-to-c                      |                                                        |
                     v                                        | }                                                      |
      +-------------------------+                             +--------------------------------------------------------+
      |  **Final Artifact A:** |                                                              |
      |                         |                                                              | Pass: -lower-hops-to-bds
      |            |                                                              v
      +-------------------------+                             +--------------------------------------------------------+
                                                              | **Stage B.2: Hardware Configuration Layer (dma_bd)** |
                                                              |--------------------------------------------------------|
                                                              | // dma_hop is replaced by a dma_bd instruction sequence. |
                                                              | dma_bd.config_bd %bd, %src, %dst, %len                 |
                                                              | dma_bd.submit %channel, %bd                            |
                                                              | dma_bd.sync %channel                                   |
                                                              | // ... linalg ...                                      |
                                                              +--------------------------------------------------------+
                                                                                             |
                                                                                             | Passes: 
                                                                                             |         
                                                                                             v
                                                                              +---------------------------+
                                                                              |   **Final Artifact B:** |
                                                                              |                           |
                                                                              |        main.exe           |
                                                                              +---------------------------+

```

