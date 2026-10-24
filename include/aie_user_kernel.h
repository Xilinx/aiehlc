/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * aie_user_kernel.h -- the C-level contract for a USER AIE kernel that replaces
 * one or more layers of a TVM-compiled network.
 *
 * A user kernel is one aiehlc source file (library mode: no main()) that
 *
 *   1. defines the AIE kernel and its mesh launch, exactly like
 *      src/aietensorop/conv2dstem/conv2dstem.cc;
 *   2. defines ONE entry with the fixed signature below:
 *
 *          AIEK_ENTRY(my_kernel) { ... return AIEK_OK; }
 *
 *      `args` are the layer's tensors in TVM's call order -- every input
 *      (activations and constant parameters, interleaved exactly as the graph
 *      passes them), then every output;
 *   3. describes that signature with AIEK_SPEC(my_kernel, "...") so the Python
 *      side (frontend/tvmrelay/aiekernel) can find the layers it fits, check
 *      the parameter values, calibrate it and generate the TVM glue.
 *
 * Optionally a plain-C x86 file defines the REFERENCE of the same op,
 *
 *          AIEK_REFERENCE(my_kernel) { ... }      // defines my_kernel_ref()
 *
 * which `python -m frontend.tvmrelay.aiekernel check` compiles with the host
 * gcc and diffs against TVM's own kernel for every matched layer, and then
 * swaps into the whole network (x86, no board) to prove the logits do not
 * move. Tutorial: doc/tutorial_userkernel.md.
 *
 * What the user does NOT write: the TVM packed-call function. The generator
 * appends one `aie_<tvm symbol>()` adapter per replaced layer to aiehlc's
 * host.cc; it binds TVM's DLTensors into aiek_tensor[], re-checks dtype and
 * shape against the spec at run time, calls the entry and prints the
 * `[aie-offload] ... ENTER` / `EXIT rc= hash=` lines.
 *
 * Rules this header itself obeys and a user source must too (each one breaks
 * the aiehlc frontend SILENTLY -- `aiekernel lint` checks them):
 *   - the aiehlc kernel keyword and the triple-angle launch token must never
 *     appear in a comment or a string (the source rewriter matches TEXT);
 *   - nothing that reaches the launch may be declared above the kernel;
 *   - helpers the kernel body calls must be #define macros written in the
 *     .cc itself, not functions and not macros from a header.
 * Everything below is host-side only; the kernel translation unit never sees
 * this header.
 *
 * Valid C11 and C++ (the entry lives in aiehlc's C++ host.cc, the reference
 * in a C file).
 ******************************************************************************/
#ifndef AIE_USER_KERNEL_H
#define AIE_USER_KERNEL_H

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#ifdef __cplusplus
#define AIEK_EXTERN_C extern "C"
#else
#define AIEK_EXTERN_C
#endif

/* ── dtype codes: DLPack's, so a DLTensor's dtype.code copies straight in ── */
#define AIEK_INT 0
#define AIEK_UINT 1
#define AIEK_FLOAT 2

#define AIEK_MAX_NDIM 8

/* ── Return codes. Anything but AIEK_OK makes the adapter return -1 to the
 *    graph executor, after printing the code. ─────────────────────────────── */
#define AIEK_OK 0
#define AIEK_ERR_ARGS (-1)   /* wrong count / dtype / shape (adapter catches)   */
#define AIEK_ERR_PARAM (-2)  /* a parameter value the kernel cannot express     */
#define AIEK_ERR_ALLOC (-3)  /* staging allocation failed                       */
#define AIEK_ERR_KERNEL (-4) /* the AIE launch itself failed                    */

/* One argument of the layer. `data` is the graph executor's buffer
 * (64-byte aligned in graph_driver.c) and is NOT retained after return. */
typedef struct {
    void *data;
    int32_t ndim;
    uint8_t code; /* AIEK_INT / AIEK_UINT / AIEK_FLOAT */
    uint8_t bits;
    uint16_t lanes;
    const int64_t *shape;
    uint64_t nbytes;
} aiek_tensor;

/* The user's entry: `AIEK_ENTRY(name) { ... }`. extern "C" in C++, so the
 * generated adapter (and the x86 reference harness) can call it by name. */
#define AIEK_ENTRY(name) AIEK_EXTERN_C int name(const aiek_tensor *args, int num_args)

/* The x86 reference of the same op: `AIEK_REFERENCE(name) { ... }` defines
 * name_ref(). Same arguments, same order, same results -- bit for bit unless
 * the spec gives the output a `tol=`. */
#define AIEK_REFERENCE(name) AIEK_EXTERN_C int name##_ref(const aiek_tensor *args, int num_args)

/* The machine-readable signature, as adjacent string literals:
 *
 *   AIEK_SPEC(my_kernel,
 *       "op        conv2d\n"
 *       "match     contrib_conv2d_NCHWc\n"
 *       "reference my_kernel_ref.c\n"
 *       "in    int8  [1,1,230,230,3]   activation\n"
 *       "param int8  [16,1,7,7,3,4]    weight\n"
 *       "param int32 [1,16,1,1,4]      zero_point_b  require=zero\n"
 *       "out   uint8 [1,16,112,112,4]  output\n");
 *
 * One line per argument, in TVM call order: kind (in | param | out), dtype,
 * shape, optional name, optional key=value (require=zero|nonzero|uniform|
 * nonneg|range(lo,hi), tol=N on outputs). A dim may be an integer, `*`, or a
 * NAME that must take the same value everywhere it appears. Grammar in
 * frontend/tvmrelay/aiekernel/spec.py. The array is kept (`used`) so the spec
 * also travels inside the archive (`strings libaiek_<name>.a`). */
#define AIEK_SPEC(entry, text) static const char aiek_spec_##entry[] __attribute__((used)) = text

/* ── Small helpers, all host-side ──────────────────────────────────────── */

#define AIEK_PTR(t, type) ((type *)(t)->data)

static inline int64_t aiek_numel(const aiek_tensor *t) {
    int64_t n = 1;
    for (int32_t i = 0; i < t->ndim; i++)
        n *= t->shape[i];
    return n;
}

/* Element i of an int32 array that may sit at ANY byte address (a layeriohex
 * `unsigned char _data[]`, a packed window). Assembled through memcpy: a
 * plain (const int32_t *) cast is an unaligned load and faults on the A78. */
static inline int32_t aiek_rd_i32(const void *base, int64_t i) {
    int32_t v;
    memcpy(&v, (const uint8_t *)base + 4 * i, sizeof v);
    return v;
}

/* Rolling hash the adapter prints on EXIT -- the same `h = h*31 + byte` the
 * stem entry uses, so a board log and a layeriohex `_out.h` compare directly. */
static inline uint32_t aiek_hash_bytes(const void *p, uint64_t n, uint32_t h) {
    const uint8_t *b = (const uint8_t *)p;
    for (uint64_t i = 0; i < n; i++)
        h = h * 31u + b[i];
    return h;
}

/* ── TVM side: what the generated adapter uses ─────────────────────────────
 *
 * The graph executor passes `void *args` = a TVMValue[] whose v_handle is a
 * DLTensor*. Mirrored here field for field (DLPack) so this header needs no
 * TVM include; the static checks below pin the layout. */
typedef struct {
    void *data;
    int32_t device_type;
    int32_t device_id;
    int32_t ndim;
    uint8_t code;
    uint8_t bits;
    uint16_t lanes;
    int64_t *shape;
    int64_t *strides;
    uint64_t byte_offset;
} aiek_dltensor;

#ifdef __cplusplus
static_assert(sizeof(aiek_dltensor) == 48, "aiek_dltensor must mirror DLTensor");
#else
_Static_assert(sizeof(aiek_dltensor) == 48, "aiek_dltensor must mirror DLTensor");
#endif

/* Bind TVM's packed args into `out[0..num_args)`. Strides must be NULL (dense):
 * the graph executor never sets them, and a kernel that silently assumed
 * dense data on a strided view would compute the wrong elements. */
static inline int aiek_bind(void *tvm_args, int num_args, aiek_tensor *out, int capacity, const char *who) {
    if (num_args != capacity) {
        printf("[aiek] %s: TVM passed %d args, the spec has %d\n", who, num_args, capacity);
        return AIEK_ERR_ARGS;
    }
    for (int i = 0; i < num_args; i++) {
        const aiek_dltensor *d = ((const aiek_dltensor *const *)tvm_args)[i];
        if (d->strides) {
            printf("[aiek] %s: arg %d is strided; only dense tensors are supported\n", who, i);
            return AIEK_ERR_ARGS;
        }
        out[i].data = (uint8_t *)d->data + d->byte_offset;
        out[i].ndim = d->ndim;
        out[i].code = d->code;
        out[i].bits = d->bits;
        out[i].lanes = d->lanes;
        out[i].shape = d->shape;
        out[i].nbytes = (uint64_t)aiek_numel(&out[i]) * ((d->bits * d->lanes + 7u) / 8u);
    }
    return AIEK_OK;
}

/* Check one bound arg against the spec. `shape` entries < 0 match any extent
 * (the spec's `*` and symbolic dims; the generator has already proved those
 * consistent for every layer it wires). */
static inline int aiek_expect(const aiek_tensor *t, int idx, uint8_t code, uint8_t bits, int32_t ndim,
                              const int64_t *shape, const char *who) {
    int ok = t->code == code && t->bits == bits && t->lanes == 1 && t->ndim == ndim;
    for (int32_t i = 0; ok && i < ndim; i++)
        ok = shape[i] < 0 || t->shape[i] == shape[i];
    if (!ok)
        printf("[aiek] %s: arg %d is code=%d bits=%d ndim=%d, the spec wants code=%d bits=%d ndim=%d\n", who, idx,
               (int)t->code, (int)t->bits, (int)t->ndim, (int)code, (int)bits, (int)ndim);
    return ok ? AIEK_OK : AIEK_ERR_ARGS;
}

#endif /* AIE_USER_KERNEL_H */
