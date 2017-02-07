/*
 * Copyright (c) 2009-2016 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "parsec.h"
#include "parsec/data_distribution.h"
#include "parsec/arena.h"

#if defined(PARSEC_HAVE_MPI)
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
 * @return the parsec object to schedule.
 */
parsec_handle_t *rtt_new(parsec_ddesc_t *A, int size, int nb)
{
    int worldsize;
    parsec_rtt_handle_t *o = NULL;
#if defined(PARSEC_HAVE_MPI)
    MPI_Comm_size(MPI_COMM_WORLD, &worldsize);
#else
    worldsize = 1;
#endif

    if( nb <= 0 || size <= 0 ) {
        fprintf(stderr, "To work, RTT must do at least one round time trip of at least one byte\n");
        return (parsec_handle_t*)o;
    }

    o = parsec_rtt_new(A, nb, worldsize);

#if defined(PARSEC_HAVE_MPI)
    {
        MPI_Aint extent;
    	MPI_Type_contiguous(size, MPI_BYTE, &block);
        MPI_Type_commit(&block);
#if defined(PARSEC_HAVE_MPI_20)
        MPI_Aint lb = 0; 
        MPI_Type_get_extent(block, &lb, &extent);
#else
        MPI_Type_extent(block, &extent);
#endif  /* defined(PARSEC_HAVE_MPI_20) */
        parsec_arena_construct(o->arenas[PARSEC_rtt_DEFAULT_ARENA],
                              extent, PARSEC_ARENA_ALIGNMENT_SSE,
                              block);
    }
#endif

    return (parsec_handle_t*)o;
}

/**
 * @param [INOUT] o the parsec object to destroy
 */
void rtt_destroy(parsec_handle_t *o)
{
#if defined(PARSEC_HAVE_MPI)
    MPI_Type_free( &block );
#endif

    PARSEC_INTERNAL_HANDLE_DESTRUCT(o);
}
