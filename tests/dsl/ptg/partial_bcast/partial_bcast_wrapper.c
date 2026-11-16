/*
 * Copyright (c) 2026      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "parsec/runtime.h"
#include "parsec/arena.h"
#include "parsec/datatype.h"
#include "parsec/data_dist/matrix/matrix.h"

#include "partial_bcast_wrapper.h"
#include "partial_bcast.h"

int32_t partial_bcast_failures = 0;

/**
 * @param [IN] A     one row of nb x nb tiles, one tile per process; rank 0
 *                   holds the tile the roots send from
 * @param [IN] nb    tile size
 * @param [IN] nc    consumers per case, one per process other than rank 0
 * @param [IN] nodd  bytes of the short payloads (below the short limit, odd)
 * @param [IN] nbig  bytes of the rendezvous payloads (above the short limit)
 * @param [IN] nrb   elements of the resized-byte receive
 * @param [IN] mask  PB_CASE_* bits of the cases to run
 *
 * @return the parsec object to schedule.
 */
parsec_taskpool_t *partial_bcast_new(parsec_matrix_block_cyclic_t *A, int nb, int nc,
                                     int nodd, int nbig, int nrb, int mask)
{
    parsec_partial_bcast_taskpool_t *tp;
    parsec_datatype_t col, trans, rb;

    tp = parsec_partial_bcast_new(A, nb, nc, nodd, nbig, nrb, mask);

    /* The tile every root sends from, and the default receive type. */
    parsec_matrix_arena_datatype_define_type(&tp->arenas_datatypes[PARSEC_partial_bcast_DEFAULT_ADT_IDX],
                                             parsec_datatype_double_t, PARSEC_MATRIX_FULL, 0,
                                             nb, nb, nb, PARSEC_ARENA_ALIGNMENT_SSE, -1);
    /* The lower triangle with its diagonal: not contiguous in order. */
    parsec_matrix_arena_datatype_define_type(&tp->arenas_datatypes[PARSEC_partial_bcast_LOWER_ADT_IDX],
                                             parsec_datatype_double_t, PARSEC_MATRIX_LOWER, 1,
                                             nb, nb, nb, PARSEC_ARENA_ALIGNMENT_SSE, -1);
    /* A transposed tile: an hvector of strided columns. No holes, lb 0 and
     * extent == size, yet it visits memory out of order. */
    parsec_type_create_vector(nb, 1, nb, parsec_datatype_double_t, &col);
    parsec_type_create_hvector(nb, 1, sizeof(double), col, &trans);
    parsec_type_free(&col);
    parsec_arena_datatype_set_type(&tp->arenas_datatypes[PARSEC_partial_bcast_TRANS_ADT_IDX],
                                   (size_t)nb * nb * sizeof(double), PARSEC_ARENA_ALIGNMENT_SSE, trans);
    /* Plain doubles in an 8-byte arena: a receive of nb*nb of them spans
     * nb*nb arena elements. */
    parsec_arena_datatype_set_type(&tp->arenas_datatypes[PARSEC_partial_bcast_DBL_ADT_IDX],
                                   sizeof(double), PARSEC_ARENA_ALIGNMENT_SSE, parsec_datatype_double_t);
    /* A byte resized to an 8-byte extent: one byte of data per 8 bytes of span. */
    parsec_type_create_resized(parsec_datatype_int8_t, 0, 8, &rb);
    parsec_arena_datatype_set_type(&tp->arenas_datatypes[PARSEC_partial_bcast_RB_ADT_IDX],
                                   8, PARSEC_ARENA_ALIGNMENT_SSE, rb);

    return (parsec_taskpool_t*)tp;
}

static void
__parsec_taskpool_partial_bcast_destructor(parsec_partial_bcast_taskpool_t *tp)
{
    parsec_matrix_arena_datatype_destruct_free_type(&tp->arenas_datatypes[PARSEC_partial_bcast_DEFAULT_ADT_IDX]);
    parsec_matrix_arena_datatype_destruct_free_type(&tp->arenas_datatypes[PARSEC_partial_bcast_LOWER_ADT_IDX]);
    parsec_matrix_arena_datatype_destruct_free_type(&tp->arenas_datatypes[PARSEC_partial_bcast_TRANS_ADT_IDX]);
    /* DBL uses a predefined type, which is not freed. */
    PARSEC_OBJ_DESTRUCT(&tp->arenas_datatypes[PARSEC_partial_bcast_DBL_ADT_IDX]);
    parsec_matrix_arena_datatype_destruct_free_type(&tp->arenas_datatypes[PARSEC_partial_bcast_RB_ADT_IDX]);
}

PARSEC_OBJ_CLASS_INSTANCE(parsec_partial_bcast_taskpool_t, parsec_taskpool_t,
                          NULL, __parsec_taskpool_partial_bcast_destructor);
