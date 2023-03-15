/*
 * Copyright (c) 2009-2020 The University of Tennessee and The University
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

#include "branching.h"
#include "branching_wrapper.h"

static void
__parsec_taskpool_branching_destructor(parsec_branching_taskpool_t* tp)
{
    (void)tp;
}

PARSEC_OBJ_CLASS_INSTANCE(parsec_branching_taskpool_t, parsec_taskpool_t,
                          NULL, __parsec_taskpool_branching_destructor);

/**
 * @param [IN] A    the data, already distributed and allocated
 * @param [IN] size size of each local data element
 * @param [IN] nb   number of iterations
 *
 * @return the parsec object to schedule.
 */
parsec_taskpool_t *branching_new(parsec_data_collection_t *A, int size, int nb)
{
    parsec_branching_taskpool_t *tp = NULL;
    parsec_datatype_t block;

    if( nb <= 0 || size <= 0 ) {
        fprintf(stderr, "To work, BRANCHING nb and size must be > 0\n");
        return (parsec_taskpool_t*)tp;
    }

    tp = parsec_branching_new(A, nb);

    ptrdiff_t lb, extent;
    parsec_type_create_contiguous(size, parsec_datatype_int_t, &block);
    parsec_type_extent(block, &lb, &extent);

    parsec_arena_datatype_construct( &tp->arenas_datatypes[PARSEC_branching_DEFAULT_ADT_IDX],
                                     extent, PARSEC_ARENA_ALIGNMENT_SSE,
                                     block );

    return (parsec_taskpool_t*)tp;
}
