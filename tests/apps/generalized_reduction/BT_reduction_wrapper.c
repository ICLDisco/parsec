/*
 * Copyright (c) 2009-2022 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "parsec/runtime.h"
#include "parsec/data_distribution.h"
#include "parsec/data_dist/matrix/matrix.h"

#if defined(PARSEC_HAVE_MPI)
#include <mpi.h>
#endif
static parsec_datatype_t block;

#include <stdio.h>

#include "BT_reduction.h"
#include "BT_reduction_wrapper.h"
#include "parsec/data_dist/matrix/two_dim_rectangle_cyclic.h"

static void
__parsec_taskpool_BT_reduction_destruct(parsec_BT_reduction_taskpool_t *tp)
{
    parsec_type_free( &(tp->arenas_datatypes[PARSEC_BT_reduction_DEFAULT_ADT_IDX].opaque_dtt) );
}

PARSEC_OBJ_CLASS_INSTANCE(parsec_BT_reduction_taskpool_t, parsec_taskpool_t,
                          NULL, __parsec_taskpool_BT_reduction_destruct);

/**
 * @param [IN] A    the data, already distributed and allocated
 * @param [IN] nb   tile size
 * @param [IN] nt   number of tiles
 *
 * @return the parsec object to schedule.
 */
parsec_taskpool_t *BT_reduction_new(parsec_tiled_matrix_t *A, int nb, int nt)
{
    parsec_BT_reduction_taskpool_t *tp = NULL;

    tp = parsec_BT_reduction_new(A, nb, nt);

    ptrdiff_t lb, extent;
    parsec_type_create_contiguous(nb, parsec_datatype_int32_t, &block);
    parsec_type_extent(block, &lb, &extent);

    parsec_arena_datatype_construct( &tp->arenas_datatypes[PARSEC_BT_reduction_DEFAULT_ADT_IDX],
                                     extent, PARSEC_ARENA_ALIGNMENT_SSE,
                                     block);

    return (parsec_taskpool_t*)tp;
}

