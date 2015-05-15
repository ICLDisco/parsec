/*
 * Copyright (c) 2009-2015 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "dague.h"
#include <stdio.h>

#include "data_distribution.h"
#include "dague/arena.h"

#include "ep.h"
#include "ep_wrapper.h"

/**
 * @param [IN] A     the data, already distributed and allocated
 * @param [IN] nt    number of tasks at a given level
 * @param [IN] level number of levels
 *
 * @return the dague object to schedule.
 */
dague_handle_t *ep_new(dague_ddesc_t *A, int nt, int level)
{
    dague_ep_handle_t *o = NULL;

    if( nt <= 0 || level <= 0 ) {
        fprintf(stderr, "To work, EP must have at least one task to run per level\n");
        return (dague_handle_t*)o;
    }

    o = dague_ep_new(nt, level, A);

#if defined(HAVE_MPI)
    {
        MPI_Aint extent;
#if defined(HAVE_MPI_20)
        MPI_Aint lb = 0; 
        MPI_Type_get_extent(MPI_BYTE, &lb, &extent);
#else
        MPI_Type_extent(MPI_BYTE, &extent);
#endif  /* defined(HAVE_MPI_20) */
        dague_arena_construct(o->arenas[DAGUE_ep_DEFAULT_ARENA],
                              extent, DAGUE_ARENA_ALIGNMENT_SSE,
                              MPI_BYTE);
    }
#endif

    return (dague_handle_t*)o;
}

/**
 * @param [INOUT] o the dague object to destroy
 */
void ep_destroy(dague_handle_t *o)
{

    DAGUE_INTERNAL_HANDLE_DESTRUCT(o);
}
