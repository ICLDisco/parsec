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

#include "merge_sort.h"
#include "merge_sort_wrapper.h"

/**
 * @param [IN] A    the data, already distributed and allocated
 * @param [IN] nb   tile size
 * @param [IN] nt   number of tiles
 *
 * @return the parsec object to schedule.
 */
parsec_handle_t *merge_sort_new(tiled_matrix_desc_t *A, int nb, int nt)
{
    parsec_merge_sort_handle_t *o = NULL;

    o = parsec_merge_sort_new(A, nb, nt);

#if defined(PARSEC_HAVE_MPI)
    parsec_arena_construct(o->arenas[PARSEC_merge_sort_DEFAULT_ARENA],
                          nb*sizeof(int), PARSEC_ARENA_ALIGNMENT_SSE,
                          MPI_INT);
#else
    parsec_arena_construct(o->arenas[PARSEC_merge_sort_DEFAULT_ARENA],
                          nb*sizeof(int), PARSEC_ARENA_ALIGNMENT_SSE,
                          PARSEC_DATATYPE_NULL);
#endif

    return (parsec_handle_t*)o;
}

/**
 * @param [INOUT] o the parsec object to destroy
 */
void merge_sort_destroy(parsec_handle_t *o)
{
#if defined(PARSEC_HAVE_MPI)
    MPI_Type_free( &block );
#endif

    PARSEC_INTERNAL_HANDLE_DESTRUCT(o);
}
