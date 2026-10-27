/*
 * Copyright (c) 2017-2023 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * Copyright (c) 2026      NVIDIA Corporation.  All rights reserved.
 */

/* parsec things */
#include "parsec/runtime.h"

/* system and io */
#include <stdlib.h>
#include <stdio.h>

#include "tests/tests_data.h"
#include "tests/tests_timing.h"
#include "tests/tests_runtime.h"
#include "parsec/interfaces/dtd/insert_function_internal.h"
#include "parsec/utils/debug.h"

#if defined(PARSEC_HAVE_STRING_H)
#include <string.h>
#endif  /* defined(PARSEC_HAVE_STRING_H) */

/* IDs for the Arena Datatypes */
static int TILE_FULL;

int
task_rank_0( parsec_execution_stream_t *es,
             parsec_task_t *this_task )
{
    (void)es;
    int *data;

    parsec_dtd_unpack_args(this_task, &data);
    //*data *= 2;

    return PARSEC_HOOK_RETURN_DONE;
}

int
task_rank_1( parsec_execution_stream_t *es,
             parsec_task_t *this_task )
{
    (void)es;
    int *data;
    int *second_data;

    parsec_dtd_unpack_args(this_task, &data, &second_data);
    //*data += 1;

    return PARSEC_HOOK_RETURN_DONE;
}

int main(int argc, char **argv)
{
    parsec_context_t* parsec;
    int rank, world, cores = -1;
    int nb, nt, i, rc;
    parsec_tiled_matrix_t *dcA;
    parsec_arena_datatype_t *adt;

    nb = 1; /* tile_size */

    if(argv[1] != NULL){
        cores = atoi(argv[1]);
    }

    rc = parsec_tests_context_init(cores, PARSEC_TEST_THREAD_SERIALIZED,
                                   &argc, &argv,
                                   &parsec, &rank, &world);
    PARSEC_CHECK_ERROR(rc, "parsec_tests_context_init");

    nt = (world > 1) ? world : 2; /* total no. of tiles */

    parsec_taskpool_t *dtd_tp = parsec_dtd_taskpool_new();

    adt = parsec_matrix_adt_new_rect(
            parsec_datatype_int32_t, nb, 1, nb);
    parsec_dtd_attach_arena_datatype(parsec, adt, &TILE_FULL);

    /* Correctness checking */
    dcA = create_and_distribute_data(rank, world, nb, nt);
    parsec_data_collection_set_key((parsec_data_collection_t *)dcA, "A");

    parsec_data_collection_t *A = (parsec_data_collection_t *)dcA;
    parsec_dtd_data_collection_init(A);

    /* Registering the dtd_handle with PARSEC context */
    rc = parsec_context_add_taskpool( parsec, dtd_tp );
    PARSEC_CHECK_ERROR(rc, "parsec_context_add_taskpool");
    rc = parsec_context_start(parsec);
    PARSEC_CHECK_ERROR(rc, "parsec_context_start");

    for( i = 0; i < nt - 1; i++ ) {
        parsec_dtd_insert_task(dtd_tp, task_rank_0, 0, PARSEC_DEV_CPU, "task_rank_0",
                               PASSED_BY_REF, PARSEC_DTD_TILE_OF_KEY(A, A->data_key(A, i, 0)),   PARSEC_INOUT | TILE_FULL | PARSEC_AFFINITY,
                               PARSEC_DTD_ARG_END );
        parsec_dtd_insert_task(dtd_tp, task_rank_1, 0, PARSEC_DEV_CPU, "task_rank_1",
                               PASSED_BY_REF, PARSEC_DTD_TILE_OF_KEY(A, A->data_key(A, i, 0)),   PARSEC_INOUT | TILE_FULL,
                               PASSED_BY_REF, PARSEC_DTD_TILE_OF_KEY(A, A->data_key(A, i+1, 0)), PARSEC_INOUT | TILE_FULL | PARSEC_AFFINITY,
                               PARSEC_DTD_ARG_END );
    }

    /*
     * A wait boundary closes this insertion epoch. Flush all touched tiles
     * first so the final DTD users are materialized, remote updates return to
     * their owners, and the same tiles can be safely reused in the next epoch.
     */
    parsec_dtd_data_flush_all( dtd_tp, A );
    rc = parsec_taskpool_wait( dtd_tp );
    PARSEC_CHECK_ERROR(rc, "parsec_taskpool_wait");

    for( i = 0; i < nt - 1; i++ ) {
        parsec_dtd_insert_task(dtd_tp, task_rank_0, 0, PARSEC_DEV_CPU, "task_rank_0",
                               PASSED_BY_REF, PARSEC_DTD_TILE_OF_KEY(A, A->data_key(A, i, 0)),   PARSEC_INOUT | TILE_FULL | PARSEC_AFFINITY,
                               PARSEC_DTD_ARG_END );
        parsec_dtd_insert_task(dtd_tp, task_rank_1, 0, PARSEC_DEV_CPU, "task_rank_1",
                               PASSED_BY_REF, PARSEC_DTD_TILE_OF_KEY(A, A->data_key(A, i, 0)),   PARSEC_INOUT | TILE_FULL,
                               PASSED_BY_REF, PARSEC_DTD_TILE_OF_KEY(A, A->data_key(A, i+1, 0)), PARSEC_INOUT | TILE_FULL | PARSEC_AFFINITY,
                               PARSEC_DTD_ARG_END );
    }

    /*
     * Flush before the final wait for the same reason: the test intentionally
     * reuses task classes across epochs, but data-version chains still need an
     * explicit terminal user before the taskpool can quiesce cleanly.
     */
    parsec_dtd_data_flush_all( dtd_tp, A );
    rc = parsec_taskpool_wait( dtd_tp );
    PARSEC_CHECK_ERROR(rc, "parsec_taskpool_wait");
    parsec_taskpool_free( dtd_tp );

    parsec_context_wait(parsec);

    parsec_dtd_free_arena_datatype(parsec, TILE_FULL);
    parsec_dtd_data_collection_fini( A );
    free_data(dcA);

    rc = parsec_tests_context_fini(&parsec);
    PARSEC_CHECK_ERROR(rc, "parsec_tests_context_fini");

    return 0;
}
