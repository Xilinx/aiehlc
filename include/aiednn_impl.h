/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * aiednn implementation table -- the contract between the runtime
 * (src/aiednn/aiednn_rt.cc) and the list the builder generates from the
 * kernel library's AIEK_SPECs (aiegraph/dnn.py -> aiednn_impl_list.cc).
 * Users include aiednn.h, not this.
 ******************************************************************************/
#ifndef AIEDNN_IMPL_H
#define AIEDNN_IMPL_H

#include "aie_user_kernel.h"
#include "aiednn.h"

#ifdef __cplusplus
extern "C" {
#endif

/* An AIEK_SPEC dim: >= 0 fixed, AIEDNN_DIM_ANY for `*`, and a NAME is
 * AIEDNN_DIM_SYM(id): every use of one id must see the same extent. */
#define AIEDNN_DIM_ANY (-1)
#define AIEDNN_DIM_SYM(id) (-2 - (id))
#define AIEDNN_MAX_ARGS 8

typedef int (*aiednn_entry_fn)(const aiek_tensor *args, int num_args);

typedef struct {
    aiednn_data_type_t data_type;
    int ndims;
    int64_t dims[AIEDNN_MAX_NDIMS];
} aiednn_arg_spec_t;

typedef struct {
    const char *name;             /* "aie:matmul_s8" */
    aiednn_primitive_kind_t kind; /* what the AIEK_SPEC's op family maps to */
    aiednn_entry_fn entry;        /* the AIEK_ENTRY */
    int nargs;                    /* primitive order: src, weights, dst */
    aiednn_arg_spec_t args[AIEDNN_MAX_ARGS];
    int specificity;    /* fixed dims; most specific acceptor wins */
    const char *source; /* kernel source, for verbose output */
} aiednn_impl_t;

/* Generated. May be empty (count 0): then only the ref: implementations run. */
extern const aiednn_impl_t aiednn_impl_list[];
extern const int aiednn_impl_count;

#ifdef __cplusplus
}
#endif

#endif /* AIEDNN_IMPL_H */
