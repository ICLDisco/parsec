/*
 * Copyright (c) 2014-2017 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "parsec/parsec_config.h"
#include "parsec.h"
#include <stdio.h>

#include "parsec/data_distribution.h"
#include "parsec/arena.h"

#include "ep.h"
#include "ep_wrapper.h"

/**
 * @param [IN] A     the data, already distributed and allocated
 * @param [IN] nt    number of tasks at a given level
 * @param [IN] level number of levels
 *
 * @return the parsec object to schedule.
 */
parsec_taskpool_t *ep_new(parsec_data_collection_t *A, int nt, int level)
{
    parsec_ep_taskpool_t *tp = NULL;

    if( nt <= 0 || level <= 0 ) {
        fprintf(stderr, "To work, EP must have at least one task to run per level\n");
        return (parsec_taskpool_t*)tp;
    }

    tp = parsec_ep_new(nt, level, A);

#if defined(PARSEC_HAVE_MPI)
    {
        MPI_Aint extent;
#if defined(PARSEC_HAVE_MPI_20)
        MPI_Aint lb = 0; 
        MPI_Type_get_extent(MPI_BYTE, &lb, &extent);
#else
        MPI_Type_extent(MPI_BYTE, &extent);
#endif  /* defined(PARSEC_HAVE_MPI_20) */
        /* The datatype is irrelevant as the example does not do communications between nodes */
        parsec_arena_construct(tp->arenas[PARSEC_ep_DEFAULT_ARENA],
                               extent, PARSEC_ARENA_ALIGNMENT_SSE,
                               MPI_BYTE);
    }
#endif

    return (parsec_taskpool_t*)tp;
}

/**
 * @param [INOUT] o the parsec object to destroy
 */
void ep_destroy(parsec_taskpool_t *o)
{

    PARSEC_INTERNAL_TASKPOOL_DESTRUCT(o);
}
