
/*
 * Copyright (c) 2009-2021 The University of Tennessee and The University
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

static void
__parsec_rtt_taskpool_destructor(parsec_rtt_taskpool_t *rtt_tp)
{
    /* We have created our own datatype, instead of using a predefined one
     * so we need to clean up.
     */
    parsec_type_free( &(rtt_tp->arenas_datatypes[PARSEC_rtt_DEFAULT_ADT_IDX].opaque_dtt) );
}

PARSEC_OBJ_CLASS_INSTANCE(parsec_rtt_taskpool_t, parsec_taskpool_t,
                   NULL, __parsec_rtt_taskpool_destructor);

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

    parsec_arena_datatype_construct( &tp->arenas_datatypes[PARSEC_rtt_DEFAULT_ADT_IDX],
                                     extent, PARSEC_ARENA_ALIGNMENT_SSE,
                                     block );
    return (parsec_taskpool_t*)tp;
}
