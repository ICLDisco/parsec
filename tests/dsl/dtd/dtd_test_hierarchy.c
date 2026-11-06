/*
 * Copyright (c) 2017-2023 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * Copyright (c) 2026      NVIDIA CORPORATION. All rights reserved.
 */

/* parsec things */
#include "parsec/runtime.h"

/* system and io */
#include <stdlib.h>
#include <stdio.h>

#include "tests/tests_timing.h"
#include "tests/tests_data.h"
#include "parsec/interfaces/dtd/insert_function_internal.h"
#include "parsec/utils/debug.h"

#if defined(PARSEC_HAVE_STRING_H)
#include <string.h>
#endif  /* defined(PARSEC_HAVE_STRING_H) */

/* This testing shows graph pruning as well as hierarchical execution.
 * The only restriction is the parsec_taskpool_wait() before parsec_context_wait()
 */

double time_elapsed = 0.0;
double sync_time_elapsed = 0.0;

int count = 0;

static int TILE_FULL = 0;

int
test_task( parsec_execution_stream_t *es,
           parsec_task_t *this_task )
{
    int amount_of_work, i, j, bla;
    parsec_dtd_unpack_args( this_task, &amount_of_work);
    for( i = 0; i < amount_of_work; i++ ) {
        //for( j = 0; j < *amount_of_work; j++ ) {
        for( j = 0; j < 2; j++ ) {
            bla = j*2;
            bla = j + 20;
            bla = j*2+i+j+i*i;
        }
    }
    count++;
    (void)es;
    (void)bla;
    return PARSEC_HOOK_RETURN_DONE;
}

int
test_task_generator( parsec_execution_stream_t *es,
                     parsec_task_t *this_task )
{
    parsec_tiled_matrix_t *dcB, *tmp;
    int rc, amount = 0, nb, nt;
    int rank = es->virtual_process->parsec_context->my_rank;
    int world = es->virtual_process->parsec_context->nb_nodes, i;

    parsec_dtd_unpack_args( this_task, &nb, &nt, &tmp);

    dcB = create_and_distribute_empty_data(rank, world, nb, nt);
    parsec_data_collection_set_key((parsec_data_collection_t *)dcB, "B");
    parsec_data_collection_t *B = (parsec_data_collection_t *)dcB;
    parsec_dtd_data_collection_init(B);

    parsec_taskpool_t *dtd_tp = parsec_dtd_taskpool_new();
    /* Registering the dtd_handle with PARSEC context */
    rc = parsec_context_add_taskpool( es->virtual_process->parsec_context, dtd_tp );
    PARSEC_CHECK_ERROR(rc, "parsec_context_add_taskpool");

    for( i = 0; i < 100; i++ ) {
        parsec_dtd_insert_task(dtd_tp, test_task, 0, PARSEC_DEV_CPU, "Test_Task",
                               sizeof(int), &amount, PARSEC_VALUE,
                               PASSED_BY_REF, PARSEC_DTD_TILE_OF_KEY(B, rank), PARSEC_INOUT | PARSEC_AFFINITY,
                               PARSEC_DTD_ARG_END);
    }

    parsec_dtd_data_flush(dtd_tp, PARSEC_DTD_TILE_OF_KEY(B, rank));

    /* finishing all the tasks inserted, but not finishing the handle */
    rc = parsec_taskpool_wait( dtd_tp );
    PARSEC_CHECK_ERROR(rc, "parsec_taskpool_wait");

    parsec_dtd_data_collection_fini(B);
    free_data(dcB);

    parsec_taskpool_free( dtd_tp );

    count++;

    (void)es;
    return PARSEC_HOOK_RETURN_DONE;
}

int main(int argc, char ** argv)
{
    parsec_context_t* parsec;
    int rank, world, cores = -1, rc;
    parsec_arena_datatype_t *adt;

    rc = parsec_tests_context_init(cores, PARSEC_TEST_THREAD_SERIALIZED,
                                   &argc, &argv, &parsec, &rank, &world);
    PARSEC_CHECK_ERROR(rc, "parsec_tests_context_init");

    int m;
    int nb, nt;
    parsec_tiled_matrix_t *dcA;
    parsec_taskpool_t *dtd_tp;

    dtd_tp = parsec_dtd_taskpool_new();

    /* Registering the dtd_handle with PARSEC context */
    rc = parsec_context_add_taskpool( parsec, dtd_tp );
    PARSEC_CHECK_ERROR(rc, "parsec_context_add_taskpool");

    nb = 1; /* size of each tile */
    nt = world; /* total tiles */

    dcA = create_and_distribute_empty_data(rank, world, nb, nt);
    parsec_data_collection_set_key((parsec_data_collection_t *)dcA, "A");

    adt = parsec_matrix_adt_new_rect(
            parsec_datatype_int32_t, nb, 1, nb);
    parsec_dtd_attach_arena_datatype(parsec, adt, &TILE_FULL);

    parsec_data_collection_t *A = (parsec_data_collection_t *)dcA;
    parsec_dtd_data_collection_init(A);

    SYNC_TIME_START(parsec);
    rc = parsec_context_start( parsec );
    PARSEC_CHECK_ERROR(rc, "parsec_context_start");

    for( m = 0; m < nt; m++ ) {
        parsec_dtd_insert_task(dtd_tp, test_task_generator, 0, PARSEC_DEV_CPU, "Test_Task_generator",
                               sizeof(int), &nb, PARSEC_VALUE,
                               sizeof(int), &nt, PARSEC_VALUE,
                               PASSED_BY_REF, PARSEC_DTD_TILE_OF_KEY(A, m),   PARSEC_INOUT | PARSEC_AFFINITY,
                               PARSEC_DTD_ARG_END);

        parsec_dtd_data_flush(dtd_tp, PARSEC_DTD_TILE_OF_KEY(A, m));
    }

    /* finishing all the tasks inserted, but not finishing the handle */
    rc = parsec_taskpool_wait( dtd_tp );
    PARSEC_CHECK_ERROR(rc, "parsec_taskpool_wait");

    parsec_output( 0, "Successfully executed %d tasks in rank %d\n", count, parsec->my_rank );

    rc = parsec_context_wait(parsec);
    PARSEC_CHECK_ERROR(rc, "parsec_context_wait");

    SYNC_TIME_PRINT(parsec, rank, ("\n") );

    parsec_dtd_free_arena_datatype(parsec, TILE_FULL);
    parsec_dtd_data_collection_fini( A );
    free_data(dcA);

    parsec_taskpool_free( dtd_tp );

    rc = parsec_tests_context_fini(&parsec);
    PARSEC_CHECK_ERROR(rc, "parsec_tests_context_fini");

    return 0;
}
