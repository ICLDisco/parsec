/* parsec things */
#include "parsec/runtime.h"

/* system and io */
#include <stdlib.h>
#include <stdio.h>

#include "common_data.h"
#include "common_timing.h"
#include "parsec/interfaces/superscalar/insert_function_internal.h"
#include "parsec/utils/debug.h"

#if defined(PARSEC_HAVE_STRING_H)
#include <string.h>
#endif  /* defined(PARSEC_HAVE_STRING_H) */

#if defined(PARSEC_HAVE_MPI)
#include <mpi.h>
#endif  /* defined(PARSEC_HAVE_MPI) */

double time_elapsed;
double sync_time_elapsed;

int32_t global_counter;

int
task_to_check_generation(parsec_execution_stream_t *es, parsec_task_t *this_task)
{
    (void)es; (void)this_task;

    (void)parsec_atomic_fetch_inc_int32(&global_counter);

    return PARSEC_HOOK_RETURN_DONE;
}

int
task_to_check_overhead_1(parsec_execution_stream_t *es, parsec_task_t *this_task)
{
    (void)es;
    int flows;
    int *data;

    parsec_dtd_unpack_args(this_task, &flows, &data);

    return PARSEC_HOOK_RETURN_DONE;
}

int
task_to_check_overhead_2(parsec_execution_stream_t *es, parsec_task_t *this_task)
{
    (void)es;
    int flows;
    int *data;

    parsec_dtd_unpack_args(this_task, &flows, &data, &data);

    return PARSEC_HOOK_RETURN_DONE;
}

int
task_to_check_overhead_3(parsec_execution_stream_t *es, parsec_task_t *this_task)
{
    (void)es;
    int flows;
    int *data;

    parsec_dtd_unpack_args(this_task, &flows, &data, &data, &data);

    return PARSEC_HOOK_RETURN_DONE;
}

int
task_to_check_overhead_5(parsec_execution_stream_t *es, parsec_task_t *this_task)
{
    (void)es;
    int flows;
    int *data;

    parsec_dtd_unpack_args(this_task, &flows, &data, &data, &data, &data,
                           &data);

    return PARSEC_HOOK_RETURN_DONE;
}

int
task_to_check_overhead_10(parsec_execution_stream_t *es, parsec_task_t *this_task)
{
    (void)es;
    int flows;
    int *data;

    parsec_dtd_unpack_args(this_task, &flows, &data, &data, &data, &data,
                           &data, &data, &data, &data, &data, &data);

    return PARSEC_HOOK_RETURN_DONE;
}

int
task_to_check_overhead_15(parsec_execution_stream_t *es, parsec_task_t *this_task)
{
    (void)es;
    int flows;
    int *data;

    parsec_dtd_unpack_args(this_task, &flows, &data, &data, &data, &data,
                           &data, &data, &data, &data, &data, &data, &data,
                           &data, &data, &data, &data);

    return PARSEC_HOOK_RETURN_DONE;
}

int main(int argc, char ** argv)
{
    parsec_context_t* parsec;
    int rank, world, cores = -1;
    int nb, nt, rc;
    parsec_tiled_matrix_dc_t *dcA;

#if defined(PARSEC_HAVE_MPI)
    {
        int provided;
        MPI_Init_thread(&argc, &argv, MPI_THREAD_SERIALIZED, &provided);
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

    if(argv[1] != NULL){
        cores = atoi(argv[1]);
    }

    /* Creating parsec context and initializing dtd environment */
    parsec = parsec_init( cores, &argc, &argv );

    /****** Checking task generation ******/
    parsec_taskpool_t *dtd_tp = parsec_dtd_taskpool_new();

    global_counter = 0; /* this counter should, at the end, be equal to total_tasks below */
    int i, j, total_tasks = 10000;

    if( 0 == rank ) {
        parsec_output( 0, "\nChecking task generation using dtd interface. "
                       "We insert 10000 tasks and atomically increase a global counter to see if %d task executed\n\n", total_tasks );
    }

    /* Registering the dtd_handle with PARSEC context */
    rc = parsec_context_add_taskpool( parsec, dtd_tp );
    PARSEC_CHECK_ERROR(rc, "parsec_context_add_taskpool");
    rc = parsec_context_start(parsec);
    PARSEC_CHECK_ERROR(rc, "parsec_context_start");

    for( i = 0; i < total_tasks; i++ ) {
        /* This task does not have any data associated with it, so it will be inserted in all mpi processes */
        parsec_dtd_taskpool_insert_task(dtd_tp, task_to_check_generation,    0,  "sample_task",
                                        PARSEC_DTD_ARG_END);
    }

    rc = parsec_dtd_taskpool_wait( parsec, dtd_tp );
    PARSEC_CHECK_ERROR(rc, "parsec_dtd_taskpool_wait");

    if( (int)global_counter != total_tasks ) {
        parsec_fatal( "Something is wrong, all tasks were not generated correctly\n" );
    }

    if( 0 == rank ) {
        parsec_output( 0, "Tasks are being generated correctly.\n\n" );
    }

    parsec_taskpool_free( dtd_tp );
    /****** End of checking task generation ******/


    /***** Start of timing overhead of task generation ******/
    int total_flows[6] = {1, 2, 3, 5, 10, 15};
    //int total_flows[6] = {1, 0, 0, 0, 0, 0};
    total_tasks = 100000;
    //total_tasks = 1;

    if( 0 == rank ) {
        parsec_output( 0, "\nChecking time of inserting tasks. We insert %d independent tasks "
                       "and measure the time for different number flows for each task (using 1 thread).\n\n", total_tasks );
    }

    for( i = 0; i < 6; i++ ) {
        nb = 1; /* size of each tile */
        nt = total_flows[i]*total_tasks; /* total tiles */
        //nt = total_tasks; /* total tiles */

        dcA = create_and_distribute_empty_data(rank, world, nb, nt);
        parsec_data_collection_set_key((parsec_data_collection_t *)dcA, "A");

        parsec_data_collection_t *A = (parsec_data_collection_t *)dcA;
        parsec_dtd_data_collection_init(A);

        dtd_tp = parsec_dtd_taskpool_new(  );

        rc = parsec_context_add_taskpool( parsec, dtd_tp );
        PARSEC_CHECK_ERROR(rc, "parsec_context_add_taskpool");

        SYNC_TIME_START();

        if( 1 == total_flows[i] ) {
            for( j = 0; j < total_flows[i] * total_tasks; j += total_flows[i] ) {
                parsec_dtd_taskpool_insert_task(dtd_tp, task_to_check_overhead_1,  0,  "task_for_timing_overhead",
                                                sizeof(int),      &total_flows[i],      VALUE,
                                                PASSED_BY_REF,    TILE_OF_KEY(A, j),    INOUT,
                                                PARSEC_DTD_ARG_END);
                }
        } else if( 2 == total_flows[i] ) {
            for( j = 0; j < total_flows[i] * total_tasks; j += total_flows[i] ) {
                parsec_dtd_taskpool_insert_task(dtd_tp, task_to_check_overhead_2,  0,  "task_for_timing_overhead",
                                                sizeof(int),      &total_flows[i],      VALUE,
                                                PASSED_BY_REF,    TILE_OF_KEY(A, j),    INOUT,
                                                PASSED_BY_REF,    TILE_OF_KEY(A, j+1),  INOUT,
                                                PARSEC_DTD_ARG_END);
            }
        } else if( 3 == total_flows[i] ) {
            for( j = 0; j < total_flows[i] * total_tasks; j += total_flows[i] ) {
                parsec_dtd_taskpool_insert_task(dtd_tp, task_to_check_overhead_3,  0,  "task_for_timing_overhead",
                                                sizeof(int),      &total_flows[i],      VALUE,
                                                PASSED_BY_REF,    TILE_OF_KEY(A, j),    INOUT,
                                                PASSED_BY_REF,    TILE_OF_KEY(A, j+1),  INOUT,
                                                PASSED_BY_REF,    TILE_OF_KEY(A, j+2),  INOUT,
                                                PARSEC_DTD_ARG_END);
            }
        } else if( 5 == total_flows[i] ) {
            for( j = 0; j < total_flows[i] * total_tasks; j += total_flows[i] ) {
                parsec_dtd_taskpool_insert_task(dtd_tp, task_to_check_overhead_5,  0,  "task_for_timing_overhead",
                                                sizeof(int),      &total_flows[i],      VALUE,
                                                PASSED_BY_REF,    TILE_OF_KEY(A, j),    INOUT,
                                                PASSED_BY_REF,    TILE_OF_KEY(A, j+1),  INOUT,
                                                PASSED_BY_REF,    TILE_OF_KEY(A, j+2),  INOUT,
                                                PASSED_BY_REF,    TILE_OF_KEY(A, j+3),  INOUT,
                                                PASSED_BY_REF,    TILE_OF_KEY(A, j+4),  INOUT,
                                                PARSEC_DTD_ARG_END);
            }
        } else if( 10 == total_flows[i] ) {
            for( j = 0; j < total_flows[i] * total_tasks; j += total_flows[i] ) {
                parsec_dtd_taskpool_insert_task(dtd_tp, task_to_check_overhead_10,  0,  "task_for_timing_overhead",
                                                sizeof(int),      &total_flows[i],      VALUE,
                                                PASSED_BY_REF,    TILE_OF_KEY(A, j),    INOUT,
                                                PASSED_BY_REF,    TILE_OF_KEY(A, j+1),  INOUT,
                                                PASSED_BY_REF,    TILE_OF_KEY(A, j+2),  INOUT,
                                                PASSED_BY_REF,    TILE_OF_KEY(A, j+3),  INOUT,
                                                PASSED_BY_REF,    TILE_OF_KEY(A, j+4),  INOUT,
                                                PASSED_BY_REF,    TILE_OF_KEY(A, j+5),  INOUT,
                                                PASSED_BY_REF,    TILE_OF_KEY(A, j+6),  INOUT,
                                                PASSED_BY_REF,    TILE_OF_KEY(A, j+7),  INOUT,
                                                PASSED_BY_REF,    TILE_OF_KEY(A, j+8),  INOUT,
                                                PASSED_BY_REF,    TILE_OF_KEY(A, j+9),  INOUT,
                                                PARSEC_DTD_ARG_END);
            }
        } else if( 15 == total_flows[i] ) {
            for( j = 0; j < total_flows[i] * total_tasks; j += total_flows[i] ) {
                parsec_dtd_taskpool_insert_task(dtd_tp, task_to_check_overhead_15,  0,  "task_for_timing_overhead",
                                                sizeof(int),      &total_flows[i],      VALUE,
                                                PASSED_BY_REF,    TILE_OF_KEY(A, j),    INOUT,
                                                PASSED_BY_REF,    TILE_OF_KEY(A, j+1),  INOUT,
                                                PASSED_BY_REF,    TILE_OF_KEY(A, j+2),  INOUT,
                                                PASSED_BY_REF,    TILE_OF_KEY(A, j+3),  INOUT,
                                                PASSED_BY_REF,    TILE_OF_KEY(A, j+4),  INOUT,
                                                PASSED_BY_REF,    TILE_OF_KEY(A, j+5),  INOUT,
                                                PASSED_BY_REF,    TILE_OF_KEY(A, j+6),  INOUT,
                                                PASSED_BY_REF,    TILE_OF_KEY(A, j+7),  INOUT,
                                                PASSED_BY_REF,    TILE_OF_KEY(A, j+8),  INOUT,
                                                PASSED_BY_REF,    TILE_OF_KEY(A, j+9),  INOUT,
                                                PASSED_BY_REF,    TILE_OF_KEY(A, j+10), INOUT,
                                                PASSED_BY_REF,    TILE_OF_KEY(A, j+11), INOUT,
                                                PASSED_BY_REF,    TILE_OF_KEY(A, j+12), INOUT,
                                                PASSED_BY_REF,    TILE_OF_KEY(A, j+13), INOUT,
                                                PASSED_BY_REF,    TILE_OF_KEY(A, j+14), INOUT,
                                                PARSEC_DTD_ARG_END);
            }
        }
        parsec_dtd_data_flush_all( dtd_tp, A );

        /* finishing all the tasks inserted, but not finishing the handle */
        rc = parsec_dtd_taskpool_wait( parsec, dtd_tp );
        PARSEC_CHECK_ERROR(rc, "parsec_dtd_taskpool_wait");

        SYNC_TIME_PRINT(rank, ("\tNo of flows : %d \tTime for each task : %lf\n\n", total_flows[i], sync_time_elapsed/total_tasks));

        parsec_taskpool_free( dtd_tp );
        parsec_dtd_data_collection_fini( A );
        free_data(dcA);
    }

    /***** End of timing overhead of task generation ******/

    rc = parsec_context_wait(parsec);
    PARSEC_CHECK_ERROR(rc, "parsec_context_wait");

    parsec_fini(&parsec);

#ifdef PARSEC_HAVE_MPI
    MPI_Finalize();
#endif

    return 0;
}
