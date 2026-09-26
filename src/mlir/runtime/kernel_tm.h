/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

#ifndef KERNEL_TM_H
#define KERNEL_TM_H

/*
 * Tile Memory-Mapped (TM) register access from an AIE core kernel.
 *
 * WHY THIS EXISTS
 * ---------------
 * A core sees three NON-OVERLAPPING address spaces, each behind its OWN
 * interface (arch doc Table 5-5, "Memory Space address ranges"):
 *
 *     PM  0x0_0000 - 0x0_4000   program memory
 *     DM  0x4_0000 - 0x7_FFFF   data memory
 *     TM  0x8_0000 - 0xF_FFFF   tile memory-mapped registers
 *
 * and a register's core-bus address is its AXI-MM address + 0x8_0000, e.g.
 * DMA_BD0_0 is 0x1_D000 on AXI-MM and 0x9_D000 from the core.
 *
 * THE ADDRESS ALONE DOES NOT SELECT THE INTERFACE. The compiler picks the
 * interface from the pointer's memory space, which a numeric cast cannot
 * express:
 *
 *     *((volatile int *)(0x80000 + off)) = v;   // plain ST  -- stays in core
 *     TM_W(off, v);                             // ST.TM     -- processor bus
 *
 * A raw-pointer write therefore reads back correctly ON the core (the store and
 * load are coherent with each other) while XAie_Read32 from the host still sees
 * 0, because the transaction never left the core. That failure is silent: no
 * warning, no error, and a same-core readback "confirms" a write that never
 * happened.
 *
 * HOW THE SPACE IS DECLARED
 * -------------------------
 * By chess_storage(TM:<base>) on an OBJECT. That macro is
 *
 *     #define chess_storage(...) __attribute__((chessLstorage(__VA_ARGS__)))
 *
 * from the Synopsys-generated processor model shipped with Vitis
 * (aietools/data/aie_ml/lib/isg/me_chess_llvm.h). The frontend maps TM to an
 * internal address-space number that is NOT documented in any header -- do not
 * hardcode it. Naming the space symbolically keeps that choice with the
 * compiler, where it belongs.
 *
 * chess_storage cannot be applied to a pointer:
 *     error: chess_storage specifies storage linking for pointer/reference
 *            types, not allowed
 * so the anchor must be a file-scope object and accesses are formed by indexing
 * off its ADDRESS. This mirrors the vendor header
 * aietools/include/adf/aie/tile_control_aie2gen.h:43 exactly.
 *
 * USAGE
 * -----
 *     #include "kernel_tm.h"          // in the kernel .cc, at file scope
 *     TM_W(0x16000, 0x1234);          // write Mem_Spare_Reg
 *     chess_memory_fence();
 *     uint32_t v = TM_R(0x16000);     // read it back
 *
 * Offsets are AXI-MM offsets (0x16000), NOT the biased core address (0x96000):
 * the anchor's base supplies the +0x80000. Reg_DB_Offset is 0 on aie2*, so the
 * two differ by exactly TM_BASE_ADDR.
 *
 * Verify the lowering, not just the value: a correct access assembles to ST.TM
 * / LDA.TM. Grep the kernel listing --
 *     grep -cE '\.TM' <build>/obj/kernel.lst
 * Zero means the accesses degraded to plain ST/LDA and never reached the bus.
 *
 * ALTERNATIVE
 * -----------
 * adf::write(adf::reg_val{off, val}) / adf::read(off) from <adf.h> do the same
 * thing through AMD's own API (it wraps the same kind of anchor). Prefer that
 * when the kernel already depends on ADF; this header exists so a kernel can
 * reach TM without pulling in ADF, and so the rule above is written down once.
 *
 * SCOPE
 * -----
 * TM exists on AIE2 and later (__AIE_ARCH__ >= 20). On AIE1 and on host-side
 * parses the macros compile to no-ops so a shared source file still builds --
 * note this file is parsed TWICE for an aiehlc kernel (once by aiehlc's
 * host-side Clang, which has neither uint32 nor chess_storage, and once by
 * xchesscc).
 */

/* AXI-MM -> core-processor-bus bias (arch doc Table 5-5). */
#define TM_BASE_ADDR 0x80000

#if defined(__AIENGINE__) && defined(__AIE_ARCH__) && (__AIE_ARCH__ >= 20)

/*
 * The anchor. Gives the compiler an object in TM; every access derived from its
 * address inherits that space. Declared `inline` so multiple translation units
 * that include this header collapse to one definition instead of colliding at
 * link time (same reasoning as the vendor header).
 */
inline volatile unsigned int chess_storage(TM : TM_BASE_ADDR) __aiehlc_tm_anchor;

/* Pointer arithmetic is in uint32 units, hence /4: the result is the byte
 * address TM_BASE_ADDR + off. `off` must be 4-byte aligned. */
#define TM_PTR(off) (&__aiehlc_tm_anchor + ((off) / 4))
#define TM_W(off, val) (*TM_PTR(off) = (unsigned int)(val))
#define TM_R(off) (*TM_PTR(off))

#define TM_SUPPORTED 1

#else /* AIE1, or a host-side parse: compile the accesses out */

#define TM_W(off, val) ((void)0)
#define TM_R(off) (0u)
#define TM_SUPPORTED 0

#endif

/* Common TM register offsets (AXI-MM), from the vendor model. Pass these
 * straight to TM_W/TM_R. */
#define TM_MEM_SPARE_REG 0x16000 /* scratch: not touched by DMA or runtime */
#define TM_DMA_BD0_0 0x1D000     /* BD stride 0x20, 6 words each */
#define TM_DMA_S2MM_0_CTRL 0x1DE00
#define TM_DMA_S2MM_0_START_QUEUE 0x1DE04
#define TM_DMA_S2MM_STATUS_0 0x1DF00
#define TM_DMA_MM2S_STATUS_0 0x1DF10
#define TM_LOCK0_VALUE 0x1F000 /* lock stride 0x10 */
#define TM_CORE_STATUS 0x32004

#endif /* KERNEL_TM_H */
