/*
 * Copyright (c) 2009-2019 The University of Tennessee and The University
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

    parsec_type_create_contiguous(size, parsec_datatype_int8_t, &block);
    parsec_arena_datatype_construct(tp->arenas_datatypes[PARSEC_branching_DEFAULT_ARENA],
                           size * sizeof(int8_t), size * sizeof(int8_t),
                           block);

    return (parsec_taskpool_t*)tp;
}

/**
 * @param [INOUT] o the parsec object to destroy
 */
void branching_destroy(parsec_taskpool_t *o)
{
    parsec_branching_taskpool_t* tp = (parsec_branching_taskpool_t*)o;

    parsec_type_free(&tp->arenas_datatypes[PARSEC_branching_DEFAULT_ARENA]->opaque_dtt);

    parsec_taskpool_free(o);
}
