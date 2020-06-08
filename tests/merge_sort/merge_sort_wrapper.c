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

#include "merge_sort.h"
#include "merge_sort_wrapper.h"

/**
 * @param [IN] A    the data, already distributed and allocated
 * @param [IN] nb   tile size
 * @param [IN] nt   number of tiles
 *
 * @return the parsec object to schedule.
 */
parsec_taskpool_t* merge_sort_new(parsec_tiled_matrix_dc_t *A, int nb, int nt)
{
    parsec_merge_sort_taskpool_t *tp = NULL;

    tp = parsec_merge_sort_new(A, nb, nt);

    /* As the datatype is parsec_datatype_int32_t all communications to/from
     * this arena should use the count property or they will exchange a
     * single integer. */
    parsec_arena_datatype_construct( &tp->arenas_datatypes[PARSEC_merge_sort_DEFAULT_ARENA],
                                     nb*sizeof(int), PARSEC_ARENA_ALIGNMENT_SSE,
                                     parsec_datatype_int32_t);

    return (parsec_taskpool_t*)tp;
}

/**
 * @param [INOUT] o the parsec object to destroy
 */
void merge_sort_destroy(parsec_taskpool_t *tp)
{

    parsec_taskpool_free(tp);
}
