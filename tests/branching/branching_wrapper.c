/*
 * Copyright (c) 2009-2015 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "dague.h"
#include "data_distribution.h"
#include "dague/arena.h"

#if defined(HAVE_MPI)
#include <mpi.h>
static MPI_Datatype block;
#endif
#include <stdio.h>

#include "branching.h"
#include "branching_wrapper.h"

/**
 * @param [IN] A    the data, already distributed and allocated
 * @param [IN] size size of each local data element
 * @param [IN] nb   number of iterations
 *
 * @return the dague object to schedule.
 */
dague_handle_t *branching_new(dague_ddesc_t *A, int size, int nb)
{
    dague_branching_handle_t *o = NULL;

    if( nb <= 0 || size <= 0 ) {
        fprintf(stderr, "To work, BRANCHING nb and size must be > 0\n");
        return (dague_handle_t*)o;
    }

    o = dague_branching_new(A, nb);

#if defined(HAVE_MPI)
    {
        MPI_Type_vector(1, size, size, MPI_BYTE, &block);
        MPI_Type_commit(&block);
        dague_arena_construct(o->arenas[DAGUE_branching_DEFAULT_ARENA],
                              size * sizeof(char), size * sizeof(char), 
                              block);
    }
#endif

    return (dague_handle_t*)o;
}

/**
 * @param [INOUT] o the dague object to destroy
 */
void branching_destroy(dague_handle_t *o)
{
#if defined(HAVE_MPI)
    MPI_Type_free( &block );
#endif

    DAGUE_INTERNAL_HANDLE_DESTRUCT(o);
}
