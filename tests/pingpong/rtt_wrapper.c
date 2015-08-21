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

#include "rtt.h"
#include "rtt_wrapper.h"

/**
 * @param [IN] A    the data, already distributed and allocated
 * @param [IN] size size of each local data element
 * @param [IN] nb   number of iterations
 *
 * @return the dague object to schedule.
 */
dague_handle_t *rtt_new(dague_ddesc_t *A, int size, int nb)
{
    int worldsize;
    dague_rtt_handle_t *o = NULL;
#if defined(HAVE_MPI)
    MPI_Comm_size(MPI_COMM_WORLD, &worldsize);
#else
    worldsize = 1;
#endif

    if( nb <= 0 || size <= 0 ) {
        fprintf(stderr, "To work, RTT must do at least one round time trip of at least one byte\n");
        return (dague_handle_t*)o;
    }

    o = dague_rtt_new(A, nb, worldsize);

#if defined(HAVE_MPI)
    {
        MPI_Aint extent;
    	MPI_Type_contiguous(size, MPI_INT, &block);
        MPI_Type_commit(&block);
#if defined(HAVE_MPI_20)
        MPI_Aint lb = 0; 
        MPI_Type_get_extent(block, &lb, &extent);
#else
        MPI_Type_extent(block, &extent);
#endif  /* defined(HAVE_MPI_20) */
        dague_arena_construct(o->arenas[DAGUE_rtt_DEFAULT_ARENA],
                              extent, DAGUE_ARENA_ALIGNMENT_SSE,
                              block);
    }
#endif

    return (dague_handle_t*)o;
}

/**
 * @param [INOUT] o the dague object to destroy
 */
void rtt_destroy(dague_handle_t *o)
{
#if defined(HAVE_MPI)
    MPI_Type_free( &block );
#endif

    DAGUE_INTERNAL_HANDLE_DESTRUCT(o);
}
