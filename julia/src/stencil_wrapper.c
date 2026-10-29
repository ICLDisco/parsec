// Minimal C wrapper exposing stencil init/apply functions for Julia ccall
// Mirrors testing_stencil_1D.c workflow

#include <stdio.h>
#include <stdlib.h>

#include "parsec.h"
#include "parsec/data_dist/matrix/matrix.h"

// Include the official stencil internals from PaRSEC tests.
// Headers live under tests/apps/stencil (sibling of julia/).
// The build script / CMakeLists add -I ../tests/apps/stencil
// (and the JDF-generated header dir under ../build/tests/apps/stencil).
#include "stencil_internal.h"

// Expose weight_1D from stencil_internal
DTYPE * weight_1D = NULL;

/* Allocate a zeroed block-cyclic descriptor */
void *jl_block_cyclic_alloc(void)
{
    return calloc(1, sizeof(parsec_matrix_block_cyclic_t));
}

/* Allocate data buffer using PaRSEC helpers and attach to descriptor */
void *jl_block_cyclic_alloc_data(parsec_matrix_block_cyclic_t *dcA)
{
    size_t typesize = parsec_datadist_getsizeoftype(dcA->super.mtype);
    size_t total = (size_t)dcA->super.nb_local_tiles * (size_t)dcA->super.bsiz * typesize;
    void *ptr = parsec_data_allocate(total);
    dcA->mat = ptr;
    return ptr;
}

/* Free data buffer and destroy descriptor */
void jl_block_cyclic_destroy(parsec_matrix_block_cyclic_t *dcA)
{
    if (NULL != dcA) {
        if (NULL != dcA->mat) {
            parsec_data_free(dcA->mat);
            dcA->mat = NULL;
        }
        parsec_tiled_matrix_destroy((parsec_tiled_matrix_t*)dcA);
        free(dcA);
    }
}

// Initialize weight_1D exactly like testing_stencil_1D.c
void jl_init_weight_1D(int R)
{
    int jj;
    if( weight_1D != NULL ) {
        free(weight_1D);
    }
    weight_1D = (DTYPE *)malloc(sizeof(DTYPE) * (2*R+1));
    for(jj = 1; jj <= R; jj++) {
        WEIGHT_1D(jj)  = (DTYPE)(1.0/(2.0*jj*R));
        WEIGHT_1D(-jj) = -(DTYPE)(1.0/(2.0*jj*R));
    }
    WEIGHT_1D(0) = (DTYPE)1.0;
}

// Call parsec_apply with stencil_1D_init_ops
int jl_parsec_apply_init(parsec_context_t* parsec,
                         parsec_tiled_matrix_t* A,
                         int R)
{
    int r = R;
    return parsec_apply(parsec, PARSEC_MATRIX_FULL, A,
                        (parsec_tiled_matrix_unary_op_t)stencil_1D_init_ops, &r);
}

// Directly forward to parsec_stencil_1D core kernel
int jl_parsec_stencil_1D(parsec_context_t* parsec,
                         parsec_tiled_matrix_t* A,
                         int iterations,
                         int radius)
{
    return parsec_stencil_1D(parsec, A, iterations, radius);
}
