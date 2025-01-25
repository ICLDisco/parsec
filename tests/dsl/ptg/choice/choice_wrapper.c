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

#include "choice.h"
#include "choice_wrapper.h"

static void
__parsec_taskpool_choice_destructor(parsec_choice_taskpool_t *tp)
{
    /* We have created our own datatype, instead of using a predefined one
     * so we need to clean up.
     */
    parsec_type_free(&(tp->arenas_datatypes[PARSEC_choice_DEFAULT_ADT_IDX].opaque_dtt));
}

PARSEC_OBJ_CLASS_INSTANCE(parsec_choice_taskpool_t, parsec_taskpool_t,
                          NULL, __parsec_taskpool_choice_destructor);

/**
 * @param [IN] A    the data, already distributed and allocated
 * @param [IN] size size of each local data element
 * @param [IN] nb   number of iterations
 *
 * @return the parsec object to schedule.
 */
parsec_taskpool_t *choice_new(parsec_data_collection_t *A, int size, int *decision, int nb, int world)
{
    parsec_choice_taskpool_t *tp = NULL;
    parsec_datatype_t newType;

    if( nb <= 0 || size <= 0 ) {
        fprintf(stderr, "To work, CHOICE nb and size must be > 0\n");
        return (parsec_taskpool_t*)tp;
    }

    tp = parsec_choice_new(A, nb, world, decision);

    parsec_type_create_contiguous(size, parsec_datatype_uint8_t, &newType);
    parsec_arena_datatype_construct( &tp->arenas_datatypes[PARSEC_choice_DEFAULT_ADT_IDX],
                                     size * sizeof(char), size * sizeof(char), newType );


    return (parsec_taskpool_t*)tp;
}

