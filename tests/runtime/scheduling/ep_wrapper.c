/*
 * Copyright (c) 2014-2021 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * Copyright (c) 2026      NVIDIA Corporation.  All rights reserved.
 */

#include "parsec/runtime.h"
#include <stdio.h>

#include "parsec/data_distribution.h"
#include "parsec/arena.h"
#include "parsec/datatype.h"

#include "ep.h"
#include "ep_wrapper.h"

/**
 * @param [IN] A     the data, already distributed and allocated
 * @param [IN] nt    number of tasks at a given level
 * @param [IN] level number of levels
 *
 * @return the parsec object to schedule.
 */
parsec_taskpool_t *ep_new(parsec_data_collection_t *A, int nt, int level)
{
    parsec_ep_taskpool_t *tp = NULL;

    if( nt <= 0 || level <= 0 ) {
        fprintf(stderr, "To work, EP must have at least one task to run per level\n");
        return (parsec_taskpool_t*)tp;
    }

    tp = parsec_ep_new(nt, level, A);

    /* The datatype is irrelevant as the example does not communicate data,
     * but use the PaRSEC datatype API so the test is not tied to MPI.
     */
    {
        ptrdiff_t lb, extent;
        int rc = parsec_type_extent(parsec_datatype_uint8_t, &lb, &extent);
        if( PARSEC_SUCCESS != rc ) {
            parsec_taskpool_free((parsec_taskpool_t*)tp);
            return NULL;
        }
        parsec_arena_datatype_set_type( &tp->arenas_datatypes[PARSEC_ep_DEFAULT_ADT_IDX],
                                         (size_t)extent, PARSEC_ARENA_ALIGNMENT_SSE,
                                         parsec_datatype_uint8_t );
    }

    return (parsec_taskpool_t*)tp;
}

void ep_free(parsec_taskpool_t *tp)
{
    parsec_ep_taskpool_t *ep_tp = (parsec_ep_taskpool_t*)tp;
    PARSEC_OBJ_DESTRUCT(&ep_tp->arenas_datatypes[PARSEC_ep_DEFAULT_ADT_IDX]);
    parsec_taskpool_free(tp);
}
