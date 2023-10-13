/*
 * Copyright (c) 2009-2023 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */
#include "parsec/runtime.h"

/* system and io */
#include <stdlib.h>
#include <stdio.h>

#include "tests/tests_data.h"
#include "tests/tests_timing.h"
#include "parsec/interfaces/dtd/insert_function_internal.h"
#include "parsec/utils/debug.h"

#if defined(PARSEC_HAVE_STRING_H)
#include <string.h>
#endif  /* defined(PARSEC_HAVE_STRING_H) */

#if defined(PARSEC_HAVE_MPI)
#include <mpi.h>
#endif  /* defined(PARSEC_HAVE_MPI) */

double time_elapsed;
double sync_time_elapsed;

/* IDs for the Arena Datatypes */
static int TILE_FULL;

int
task_for_timing_0( parsec_execution_stream_t *es,
                   parsec_task_t *this_task )
{
    (void)es; (void)this_task;

    return PARSEC_HOOK_RETURN_DONE;
}

int
task_for_timing_1( parsec_execution_stream_t *es,
                   parsec_task_t *this_task )
{
    (void)es; (void)this_task;

    return PARSEC_HOOK_RETURN_DONE;
}

int
task_rank_0( parsec_execution_stream_t  *es,
             parsec_task_t *this_task )
{
    (void)es;
    int *data;

    parsec_dtd_unpack_args(this_task, &data);
    *data *= 2;

    return PARSEC_HOOK_RETURN_DONE;
}

int
task_rank_1( parsec_execution_stream_t  *es,
             parsec_task_t *this_task )
{
    (void)es;
    int *data;
    int *second_data;

    parsec_dtd_unpack_args(this_task, &data, &second_data);
    *data += 1;

    return PARSEC_HOOK_RETURN_DONE;
}

int main(int argc, char **argv)
{
    parsec_context_t* parsec;
    int rank, world, cores = -1;
    int nb, nt, rc;
    parsec_tiled_matrix_t *dcA;
    parsec_arena_datatype_t *adt;

#if defined(PARSEC_HAVE_MPI)
    {
        int provided;
        MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
        if(MPI_THREAD_MULTIPLE > provided) {
            parsec_fatal( "This benchmark requires MPI_THREAD_MULTIPLE because it uses simultaneously MPI within the PaRSEC runtime, and in the main program loop (in SYNC_TIME_START)");
        }
    }
    MPI_Comm_size(MPI_COMM_WORLD, &world);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#else
    world = 1;
    rank = 0;
#endif

    if( world != 2 ) {
        parsec_fatal( "Nope! world is not right, we need exactly two MPI process. "
                      "Try with \"mpirun -np 2 .....\"\n" );
    }

    nb = 1; /* tile_size */
    nt = 2; /* total no. of tiles */

    if(argv[1] != NULL){
        cores = atoi(argv[1]);
    }

    parsec = parsec_init( cores, &argc, &argv );

    parsec_taskpool_t *dtd_tp = parsec_dtd_taskpool_new();

    adt = parsec_dtd_create_arena_datatype(parsec, &TILE_FULL);
    parsec_add2arena_rect( adt,
                                  parsec_datatype_int32_t,
                                  nb, 1, nb );

    /* Correctness checking */
    dcA = create_and_distribute_data(rank, world, nb, nt);
    parsec_data_collection_set_key((parsec_data_collection_t *)dcA, "A");

    parsec_data_collection_t *A = (parsec_data_collection_t *)dcA;
    parsec_dtd_data_collection_init(A);

    if( 0 == rank ) {
        parsec_output( 0, "\nChecking correctness of pingpong. We send data from rank 0 to rank 1 "
                       "And vice versa.\nWe start with 0 as data and should end up with 1 after "
                       "the trip.\n\n" );
    }

    parsec_data_copy_t *gdata;
    parsec_data_t *data;
    int *real_data, key;

    if( 0 == rank ) {
        key = A->data_key(A, 0, 0);
        data = A->data_of_key(A, key);
        gdata = data->device_copies[0];
        real_data = PARSEC_DATA_COPY_GET_PTR((parsec_data_copy_t *) gdata);
        *real_data = 0;
        parsec_output( 0, "Node: %d A At key[%d]: %d\n", rank, key, *real_data );
    }

    /* Registering the dtd_handle with PARSEC context */
    rc = parsec_context_add_taskpool( parsec, dtd_tp );
    PARSEC_CHECK_ERROR(rc, "parsec_context_add_taskpool");
    rc = parsec_context_start(parsec);
    PARSEC_CHECK_ERROR(rc, "parsec_context_start");

    parsec_dtd_insert_task(dtd_tp, task_rank_0, 0, PARSEC_DEV_CPU, "task_rank_0",
                           PASSED_BY_REF, PARSEC_DTD_TILE_OF_KEY(A, 0), PARSEC_INOUT | TILE_FULL | PARSEC_AFFINITY,
                           PARSEC_DTD_ARG_END);
    parsec_dtd_insert_task(dtd_tp, task_rank_1, 0, PARSEC_DEV_CPU, "task_rank_1",
                           PASSED_BY_REF, PARSEC_DTD_TILE_OF_KEY(A, 0), PARSEC_INOUT | TILE_FULL,
                           PASSED_BY_REF, PARSEC_DTD_TILE_OF_KEY(A, 1), PARSEC_INOUT | TILE_FULL | PARSEC_AFFINITY,
                           PARSEC_DTD_ARG_END);

    parsec_dtd_data_flush_all( dtd_tp, A );

    rc = parsec_taskpool_wait( dtd_tp );
    PARSEC_CHECK_ERROR(rc, "parsec_taskpool_wait");
    rc = parsec_context_wait(parsec);
    PARSEC_CHECK_ERROR(rc, "parsec_context_wait");

    if( 0 == rank ) {
        key = A->data_key(A, 0, 0);
        data = A->data_of_key(A, key);
        gdata = data->device_copies[0];
        real_data = PARSEC_DATA_COPY_GET_PTR((parsec_data_copy_t *) gdata);
        parsec_output( 0, "Node: %d A At key[%d]: %d\n", rank, key, *real_data );
        assert( *real_data == 1);
    }

    parsec_del2arena(adt);
    PARSEC_OBJ_RELEASE(adt->arena);

    parsec_dtd_data_collection_fini( A );
    free_data(dcA);

    if( 0 == rank ) {
        parsec_output( 0, "\nPingpong is behaving correctly.\n" );
    }

    parsec_taskpool_free( dtd_tp );

    /* End of correctness checking */


    /* Start of Pingpong timing */
    int repeat_pingpong = 1;
    if( 0 == rank ) {
        parsec_output( 0, "\nChecking time of pingpong. We send data from rank 0 to rank 1 "
            "And vice versa.\nWe perform this pingpong for %d times and measure the time. "
            "We report the time for different size of data for each trip.\n\n", repeat_pingpong );
    }

    int sizes_of_data = 4, i, j;
    int sizes[4] = {100, 1000, 10000, 100000};


    for( i = 0; i < sizes_of_data; i++ ) {
        dtd_tp = parsec_dtd_taskpool_new();

        rc = parsec_context_add_taskpool( parsec, dtd_tp );
        PARSEC_CHECK_ERROR(rc, "parsec_context_add_taskpool");
        rc = parsec_context_start(parsec);
        PARSEC_CHECK_ERROR(rc, "parsec_context_start");

        nb = sizes[i];
        nt = 2;

        parsec_add2arena_rect( adt,
                                      parsec_datatype_int32_t,
                                      nb, 1, nb);

        dcA = create_and_distribute_data(rank, world, nb, nt);
        parsec_data_collection_set_key((parsec_data_collection_t *)dcA, "A");

        parsec_data_collection_t *A = (parsec_data_collection_t *)dcA;
        parsec_dtd_data_collection_init(A);

        SYNC_TIME_START();
        for( j = 0; j < repeat_pingpong; j++ ) {
            parsec_dtd_insert_task(dtd_tp, task_rank_0, 0, PARSEC_DEV_CPU, "task_for_timing_0",
                                   PASSED_BY_REF, PARSEC_DTD_TILE_OF_KEY(A, 0), PARSEC_INOUT | TILE_FULL | PARSEC_AFFINITY,
                                   PARSEC_DTD_ARG_END);
            parsec_dtd_insert_task(dtd_tp, task_rank_1, 0, PARSEC_DEV_CPU, "task_for_timing_1",
                                   PASSED_BY_REF, PARSEC_DTD_TILE_OF_KEY(A, 0), PARSEC_INOUT | TILE_FULL,
                                   PASSED_BY_REF, PARSEC_DTD_TILE_OF_KEY(A, 1), PARSEC_INOUT | TILE_FULL | PARSEC_AFFINITY,
                                   PARSEC_DTD_ARG_END);
        }
        parsec_dtd_data_flush_all( dtd_tp, A );
        /* finishing all the tasks inserted, but not finishing the handle */
        rc = parsec_taskpool_wait( dtd_tp );
        PARSEC_CHECK_ERROR(rc, "parsec_taskpool_wait");

        rc = parsec_context_wait(parsec);
        PARSEC_CHECK_ERROR(rc, "parsec_context_wait");
        SYNC_TIME_PRINT(rank, ("\tSize of message : %zu bytes\tTime for each pingpong : %12.5f\n", sizes[i]*sizeof(int), sync_time_elapsed/repeat_pingpong));

        parsec_del2arena(adt);
        PARSEC_OBJ_RELEASE(adt->arena);
        parsec_dtd_data_collection_fini( A );
        free_data(dcA);

        parsec_taskpool_free(dtd_tp);
    }

    parsec_dtd_destroy_arena_datatype(parsec, TILE_FULL);
    parsec_fini(&parsec);

#ifdef PARSEC_HAVE_MPI
    MPI_Finalize();
#endif

    return 0;
}
