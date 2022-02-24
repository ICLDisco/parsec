/*
 * Copyright (c) 2017-2021 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
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

#if defined(PARSEC_HAVE_MPI)
#include <mpi.h>
#endif  /* defined(PARSEC_HAVE_MPI) */

double time_elapsed = 0.0;
double sync_time_elapsed = 0.0;

int count = 0;

/* IDs for the Arena Datatypes */
static int TILE_FULL;

int
test_task( parsec_execution_stream_t *es,
           parsec_task_t *this_task )
{
    int amount_of_work, i, j, bla;
    void *data;

    parsec_dtd_unpack_args(this_task, &amount_of_work, &data);
    for( i = 0; i < amount_of_work; i++ ) {
        for( j = 0; j < 2; j++ ) {
            bla = j*2;
            bla = j + 20;
            bla = j*2+i+j+i*i;
        }
    }
    count++;
    (void)bla;
    (void)es;
    return PARSEC_HOOK_RETURN_DONE;
}

int
test_task_generator( parsec_execution_stream_t *es,
                     parsec_task_t *this_task )
{
    (void)es;

    parsec_tiled_matrix_t *dcA;
    parsec_taskpool_t *dtd_tp = this_task->taskpool;
    int total, step, *iteration, n;
    int amount_of_work;
    int i;

    parsec_dtd_unpack_args( this_task, &n, &amount_of_work, &total, &step, &iteration, &dcA);

    parsec_data_collection_t *A = (parsec_data_collection_t *)dcA;
    for( i = 0; *iteration < total; *iteration += 1, i++ ) {
        if( i > step ) {
            return PARSEC_HOOK_RETURN_AGAIN;
        } else {
            parsec_dtd_insert_task(dtd_tp, test_task, 0, PARSEC_DEV_CPU, "Test_Task",
                                   sizeof(int), &amount_of_work, PARSEC_VALUE,
                                   PASSED_BY_REF, PARSEC_DTD_TILE_OF_KEY(A, n), PARSEC_INOUT,
                                   PARSEC_DTD_ARG_END );
        }
    }
    parsec_dtd_data_flush(dtd_tp, PARSEC_DTD_TILE_OF_KEY(A, n));

    return PARSEC_HOOK_RETURN_DONE;
}

int main(int argc, char ** argv)
{
    parsec_context_t* parsec;
    int rank, world, cores = 8, rc;

#if defined(PARSEC_HAVE_MPI)
    {
        int provided;
        MPI_Init_thread(&argc, &argv, MPI_THREAD_SERIALIZED, &provided);
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

    if( world != 1 ) {
        parsec_fatal( "Nope! world is not right, we need exactly one MPI process. "
                      "Try with \"mpirun -np 1 .....\"\n" );
    }

    int m, n;
    int no_of_chain;
    int nb, nt;
    parsec_tiled_matrix_t *dcA;
    int amount_of_work[3] = {1000, 10000, 100000};
    parsec_taskpool_t *dtd_tp;
    parsec_arena_datatype_t *adt;

    int tasks_in_each_chain[3] = {1000, 10000, 100000};

    parsec_param_set_int("runtime_num_cores", cores);
    parsec = parsec_init( &argc, &argv );
    /* compute the number of cores as detected from parsec */
    for(int p = 0, cores = 0; p < parsec->nb_vp; p++) {
        cores += parsec->virtual_processes[p]->nb_cores;
    }
    no_of_chain = cores;

    dtd_tp = parsec_dtd_taskpool_new();

    /* Registering the dtd_taskpool with PARSEC context */
    rc = parsec_context_add_taskpool( parsec, dtd_tp );
    PARSEC_CHECK_ERROR(rc, "parsec_context_add_taskpool");
    rc = parsec_context_start( parsec );
    PARSEC_CHECK_ERROR(rc, "parsec_context_start");

    nb = 1; /* size of each tile */
    nt = no_of_chain; /* total tiles */

    dcA = create_and_distribute_data(rank, world, nb, nt);
    parsec_data_collection_set_key((parsec_data_collection_t *)dcA, "A");

    adt = parsec_dtd_create_arena_datatype(parsec, &TILE_FULL);
    parsec_add2arena_rect( adt,
                                  parsec_datatype_int32_t,
                                  nb, 1, nb);

    parsec_data_collection_t *A = (parsec_data_collection_t *)dcA;
    parsec_dtd_data_collection_init(A);
    int i;
    int work_index = 0;

    for( i = 0; i < 3; i++ ) {

        SYNC_TIME_START();
        for( n = 0; n < no_of_chain; n++ ) {
            for( m = 0; m < tasks_in_each_chain[i]; m++ ) {
                parsec_dtd_insert_task(dtd_tp, test_task, 0, PARSEC_DEV_CPU, "Test_Task",
                                       sizeof(int), &amount_of_work[work_index], PARSEC_VALUE,
                                       PASSED_BY_REF, PARSEC_DTD_TILE_OF_KEY(A, n), PARSEC_INOUT,
                                       PARSEC_DTD_ARG_END );
            }
            parsec_dtd_data_flush(dtd_tp, PARSEC_DTD_TILE_OF_KEY(A, n));
        }
        /* finishing all the tasks inserted, but not finishing the taskpool */
        rc = parsec_dtd_taskpool_wait( dtd_tp );
        PARSEC_CHECK_ERROR(rc, "parsec_dtd_taskpool_wait");

        SYNC_TIME_PRINT(rank, ("No of chains : %d, No of tasks in each chain: %d,  Amount of work: %d\n", no_of_chain, tasks_in_each_chain[i], amount_of_work[work_index]));
    }

    count = 0;
    for( i = 0; i < 3; i++ ) {
        SYNC_TIME_START();
        int step = parsec_dtd_window_size, iteration = 0;

        for( n = 0; n < no_of_chain; n++ ) {
            parsec_dtd_insert_task(dtd_tp, test_task_generator, 0, PARSEC_DEV_CPU, "Test_Task_Generator",
                                   sizeof(int), &n, PARSEC_VALUE,
                                   sizeof(int), &amount_of_work[work_index], PARSEC_VALUE,
                                   sizeof(int), &tasks_in_each_chain[i], PARSEC_VALUE,
                                   sizeof(int), &step, PARSEC_VALUE,
                                   sizeof(int), &iteration, PARSEC_REF,
                                   sizeof(parsec_tiled_matrix_t*), dcA, PARSEC_REF,
                                   PARSEC_DTD_ARG_END );
        }

        /* finishing all the tasks inserted, but not finishing the taskpool */
        rc = parsec_dtd_taskpool_wait( dtd_tp );
        PARSEC_CHECK_ERROR(rc, "parsec_dtd_taskpool_wait");

        SYNC_TIME_PRINT(rank, ("No of chains : %d, No of tasks in each chain: %d,  Amount of work: %d\n", no_of_chain, tasks_in_each_chain[i], amount_of_work[work_index]));
    }
    rc = parsec_context_wait(parsec);
    PARSEC_CHECK_ERROR(rc, "parsec_context_wait");

    parsec_del2arena(adt);
    PARSEC_OBJ_RELEASE(adt->arena);
    parsec_dtd_destroy_arena_datatype(parsec, TILE_FULL);
    parsec_dtd_data_collection_fini( A );
    free_data(dcA);

    parsec_taskpool_free( dtd_tp );

    parsec_fini(&parsec);

#ifdef PARSEC_HAVE_MPI
    MPI_Finalize();
#endif

    return 0;
}
