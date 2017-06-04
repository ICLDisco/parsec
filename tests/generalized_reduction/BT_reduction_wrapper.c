/*
 * Copyright (c) 2009-2017 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "parsec_config.h"
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
parsec_handle_t *BT_reduction_new(tiled_matrix_desc_t *A, int nb, int nt)
{
    parsec_BT_reduction_handle_t *o = NULL;

    o = parsec_BT_reduction_new(A, nb, nt);

#if defined(PARSEC_HAVE_MPI)
    {
        MPI_Aint extent;
        MPI_Type_contiguous(nb, MPI_INT, &block);
        MPI_Type_commit(&block);
#if defined(PARSEC_HAVE_MPI_20)
        MPI_Aint lb = 0;
        MPI_Type_get_extent(block, &lb, &extent);
#else
        MPI_Type_extent(block, &extent);
#endif  /* defined(PARSEC_HAVE_MPI_20) */
        parsec_arena_construct(o->arenas[PARSEC_BT_reduction_DEFAULT_ARENA],
                              extent, PARSEC_ARENA_ALIGNMENT_SSE,
                              block);
    }
#endif

    return (parsec_handle_t*)o;
}

/**
 * @param [INOUT] o the parsec object to destroy
 */
void BT_reduction_destroy(parsec_handle_t *o)
{
#if defined(PARSEC_HAVE_MPI)
    MPI_Type_free( &block );
#endif

    PARSEC_INTERNAL_HANDLE_DESTRUCT(o);
}
