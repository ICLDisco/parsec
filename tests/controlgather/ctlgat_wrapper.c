/*
 * Copyright (c) 2009-2018 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "parsec.h"
#include "parsec/data_distribution.h"
#include "parsec/arena.h"

#if defined(PARSEC_HAVE_MPI)
#include <mpi.h>
#endif
#include <stdio.h>

#include "ctlgat.h"
#include "ctlgat_wrapper.h"

static parsec_datatype_t block;

/**
 * @param [IN] A    the data, already distributed and allocated
 * @param [IN] size size of each local data element
 * @param [IN] nb   number of iterations
 *
 * @return the parsec object to schedule.
 */
parsec_taskpool_t *ctlgat_new(parsec_data_collection_t *A, int size, int nb)
{
    int worldsize;
    parsec_ctlgat_taskpool_t *tp = NULL;
#if defined(PARSEC_HAVE_MPI)
    MPI_Comm_size(MPI_COMM_WORLD, &worldsize);
#else
    worldsize = 1;
#endif

    if( nb <= 0 || size <= 0 ) {
        fprintf(stderr, "To work, CTLGAT must do at least one round time trip of at least one byte\n");
        return (parsec_taskpool_t*)tp;
    }

    tp = parsec_ctlgat_new(A, nb, worldsize);

    parsec_type_create_contiguous(size, parsec_datatype_uint8_t, &block);
    parsec_arena_construct(tp->arenas[PARSEC_ctlgat_DEFAULT_ARENA],
                           size * sizeof(uint8_t), PARSEC_ARENA_ALIGNMENT_SSE,
                           block);

    return (parsec_taskpool_t*)tp;
}

/**
 * @param [INOUT] o the parsec object to destroy
 */
void ctlgat_destroy(parsec_taskpool_t *tp)
{
    parsec_type_free( &block );

    PARSEC_INTERNAL_TASKPOOL_DESTRUCT(tp);
}
