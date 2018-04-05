/*
 * Copyright (c) 2009-2018 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "parsec/parsec_config.h"
#include "parsec.h"
#include "parsec/data_distribution.h"
#include "parsec/arena.h"

#if defined(PARSEC_HAVE_MPI)
#include <mpi.h>
static MPI_Datatype block;
#endif
#include <stdio.h>

#include "BT_reduction.h"
#include "BT_reduction_wrapper.h"

/**
 * @param [IN] A    the data, already distributed and allocated
 * @param [IN] nb   tile size
 * @param [IN] nt   number of tiles
 *
 * @return the parsec object to schedule.
 */
parsec_taskpool_t *BT_reduction_new(parsec_tiled_matrix_dc_t *A, int nb, int nt)
{
    parsec_BT_reduction_taskpool_t *tp = NULL;

    tp = parsec_BT_reduction_new(A, nb, nt);

    ptrdiff_t lb, extent;
    parsec_type_create_contiguous(nb, MPI_INT, &block);
    parsec_type_extent(block, &lb, &extent);

    parsec_arena_construct(tp->arenas[PARSEC_BT_reduction_DEFAULT_ARENA],
                           extent, PARSEC_ARENA_ALIGNMENT_SSE,
                           block);

    return (parsec_taskpool_t*)tp;
}

/**
 * @param [INOUT] o the parsec object to destroy
 */
void BT_reduction_destroy(parsec_taskpool_t *o)
{
#if defined(PARSEC_HAVE_MPI)
    MPI_Type_free( &block );
#endif

    PARSEC_INTERNAL_TASKPOOL_DESTRUCT(o);
}
