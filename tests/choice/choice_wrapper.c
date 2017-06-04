/*
 * Copyright (c) 2009-2017 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "parsec_config.h"
#include "parsec.h"
#include "parsec/data_distribution.h"
#include "parsec/arena.h"

#if defined(PARSEC_HAVE_MPI)
#include <mpi.h>
static MPI_Datatype block;
#endif
#include <stdio.h>

#include "choice.h"
#include "choice_wrapper.h"

/**
 * @param [IN] A    the data, already distributed and allocated
 * @param [IN] size size of each local data element
 * @param [IN] nb   number of iterations
 *
 * @return the parsec object to schedule.
 */
parsec_handle_t *choice_new(parsec_ddesc_t *A, int size, int *decision, int nb, int world)
{
    parsec_choice_handle_t *o = NULL;

    if( nb <= 0 || size <= 0 ) {
        fprintf(stderr, "To work, CHOICE nb and size must be > 0\n");
        return (parsec_handle_t*)o;
    }

    o = parsec_choice_new(A, nb, world, decision);

#if defined(PARSEC_HAVE_MPI)
    {
        MPI_Type_vector(1, size, size, MPI_BYTE, &block);
        MPI_Type_commit(&block);
        parsec_arena_construct(o->arenas[PARSEC_choice_DEFAULT_ARENA],
                              size * sizeof(char), size * sizeof(char),
                              block);
    }
#endif

    return (parsec_handle_t*)o;
}

/**
 * @param [INOUT] o the parsec object to destroy
 */
void choice_destroy(parsec_handle_t *o)
{
    parsec_choice_handle_t *c = (parsec_choice_handle_t*)o;
    (void)c;

#if defined(PARSEC_HAVE_MPI)
    MPI_Type_free( &block );
#endif

    PARSEC_INTERNAL_HANDLE_DESTRUCT(o);
}
