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

    if( nb <= 0 || size <= 0 ) {
        fprintf(stderr, "To work, BRANCHING nb and size must be > 0\n");
        return (parsec_taskpool_t*)tp;
    }

    tp = parsec_branching_new(A, nb);

    return (parsec_taskpool_t*)tp;
}
