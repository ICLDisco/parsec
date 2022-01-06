/*
 * Copyright (c) 2018-2022 The University of Tennessee and The University
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

#include "check_multisize_bcast_wrapper.h"
#include "check_multisize_bcast.h"

/**
 * @param [IN] A    the data, already distributed and allocated
 * @param [IN] nb   matrix size
 * @param [IN] nt   tile size
 *
 * @return the parsec object to schedule.
 */
parsec_taskpool_t* check_multisize_bcast_new(parsec_matrix_block_cyclic_t *A, int nb, int nt)
{
    parsec_check_multisize_bcast_taskpool_t *tp = NULL;

    tp = parsec_check_multisize_bcast_new(A, nb, nt);

    /* As the datatype is parsec_datatype_int32_t all communications to/from
     * this arena should use the count property or they will exchange a
     * single integer. */
    parsec_arena_datatype_construct(&tp->arenas_datatypes[PARSEC_check_multisize_bcast_DEFAULT_ADT_IDX],
                           nb*sizeof(int), PARSEC_ARENA_ALIGNMENT_SSE,
                           parsec_datatype_int32_t);

    return (parsec_taskpool_t*)tp;
}

/**
 * @param [INOUT] o the parsec object to destroy
 */
static void
check_multisize_bcast_destructor(parsec_check_multisize_bcast_taskpool_t *tp)
{
    parsec_del2arena(&tp->arenas_datatypes[PARSEC_check_multisize_bcast_DEFAULT_ADT_IDX]);
}

PARSEC_OBJ_CLASS_INSTANCE(parsec_check_multisize_bcast_taskpool_t, parsec_taskpool_t,
                          NULL, check_multisize_bcast_destructor);
