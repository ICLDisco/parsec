/*
 * Copyright (c) 2009-2018 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "parsec/runtime.h"
#include "parsec/data_distribution.h"
#include "parsec/arena.h"

#if defined(PARSEC_HAVE_MPI)
#include <mpi.h>
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
parsec_taskpool_t *rtt_new(parsec_data_collection_t *A, int size, int nb)
{
    parsec_rtt_taskpool_t *tp = NULL;
    parsec_datatype_t block;
    int worldsize = 1;

#if defined(PARSEC_HAVE_MPI)
    MPI_Comm_size(MPI_COMM_WORLD, &worldsize);
#endif

    if( nb <= 0 || size <= 0 ) {
        fprintf(stderr, "To work, RTT must do at least one round time trip of at least one byte\n");
        return (parsec_taskpool_t*)tp;
    }

    tp = parsec_rtt_new(A, nb, worldsize);

    ptrdiff_t lb, extent;
    parsec_type_create_contiguous(size, parsec_datatype_uint8_t, &block);
    parsec_type_extent(block, &lb, &extent);

    parsec_arena_construct(tp->arenas[PARSEC_rtt_DEFAULT_ARENA],
                           extent, PARSEC_ARENA_ALIGNMENT_SSE,
                           block);

    return (parsec_taskpool_t*)tp;
}

/**
 * @param [INOUT] o the parsec object to destroy
 */
void rtt_destroy(parsec_taskpool_t *tp)
{
    parsec_rtt_taskpool_t *rtt_tp = (parsec_rtt_taskpool_t*)tp;

    parsec_type_free( &(rtt_tp->arenas[PARSEC_rtt_DEFAULT_ARENA]->opaque_dtt) );

    PARSEC_INTERNAL_TASKPOOL_DESTRUCT(tp);
}
