/*
 * Copyright (c) 2009-2015 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "dague.h"
#include "dague/data_distribution.h"
#include "dague/arena.h"

#if defined(HAVE_MPI)
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
 * @return the dague object to schedule.
 */
dague_handle_t *merge_sort_new(tiled_matrix_desc_t *A, int nb, int nt)
{
    dague_merge_sort_handle_t *o = NULL;

    o = dague_merge_sort_new(A, nb, nt);

#if defined(HAVE_MPI)
    dague_arena_construct(o->arenas[DAGUE_merge_sort_DEFAULT_ARENA],
                          nb*sizeof(int), DAGUE_ARENA_ALIGNMENT_SSE,
                          MPI_INT);
#else
    dague_arena_construct(o->arenas[DAGUE_merge_sort_DEFAULT_ARENA],
                          nb*sizeof(int), DAGUE_ARENA_ALIGNMENT_SSE,
                          DAGUE_DATATYPE_NULL);
#endif

    return (dague_handle_t*)o;
}

/**
 * @param [INOUT] o the dague object to destroy
 */
void merge_sort_destroy(dague_handle_t *o)
{
#if defined(HAVE_MPI)
    MPI_Type_free( &block );
#endif

    DAGUE_INTERNAL_HANDLE_DESTRUCT(o);
}
