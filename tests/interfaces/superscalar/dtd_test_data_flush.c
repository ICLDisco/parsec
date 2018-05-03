/* parsec things */
#include "parsec.h"

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

int
task_to_check_overhead_1(parsec_execution_stream_t *es, parsec_task_t *this_task)
{
    (void)es;
    int rank;
    int *data;

    parsec_dtd_unpack_args(this_task, &rank, &data);

    *data = 1;

    return PARSEC_HOOK_RETURN_DONE;
}

int
task_for_test_2_dist_mem(parsec_execution_stream_t *es, parsec_task_t *this_task)
{
    (void)es;
    int rank;
    int *data;

    parsec_dtd_unpack_args(this_task, &rank, &data);

    if(rank == 0) {
        *data += 10;
    } else if (rank == 1) {
        assert(*data == 10);
        *data += 1;
    }

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

    if(argv[1] != NULL){
        cores = atoi(argv[1]);
    }

    int i, j, total_tasks = 10000;

    /* Creating parsec context and initializing dtd environment */
    parsec = parsec_init(cores, &argc, &argv);
    parsec_taskpool_t *dtd_tp;

    /*
     ****************************************
            *** Shared Memory test ***
     ****************************************
     */
    if(world == 1) {
        /* Registering the dtd_handle with PARSEC context */
        rc = parsec_context_start(parsec);
        PARSEC_CHECK_ERROR(rc, "parsec_context_start");

        /***** Start of timing overhead of task generation ******/
        int total_flows[2] = {10, 15};
        total_tasks = 1;

        parsec_output( 0, "\nWe check if the test passes in shared memory with more than 10 unique flows\n");

        for( i = 0; i < 2; i++ ) {
            nb = 1; /* size of each tile */
            nt = total_flows[i]; /* total tiles */

            dcA = create_and_distribute_empty_data(rank, world, nb, nt);
            parsec_data_collection_set_key((parsec_data_collection_t *)dcA, "A");

            parsec_data_collection_t *A = (parsec_data_collection_t *)dcA;
            parsec_dtd_data_collection_init(A);

            dtd_tp = parsec_dtd_taskpool_new();

            rc = parsec_context_add_taskpool( parsec, dtd_tp );
            PARSEC_CHECK_ERROR(rc, "parsec_context_add_taskpool");

            if( 10 == total_flows[i] ) {
                for( j = 0; j < total_tasks; j += total_flows[i] ) {
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
                for( j = 0; j < total_tasks; j += total_flows[i] ) {
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

            parsec_taskpool_free( dtd_tp );
            parsec_dtd_data_collection_fini( A );
            free_data(dcA);
        }

        rc = parsec_context_wait(parsec);
        PARSEC_CHECK_ERROR(rc, "parsec_context_wait");

        parsec_output( 0, "Shared Memory test PASSED\n");

    } else if (world == 2) {
    /*
     ***************************************
         *** Distributed Memory test ***
     ***************************************
     */
        /**** We send data from rank 0 to 1 and flush it back */

        /* Registering the dtd_handle with PARSEC context */
        rc = parsec_context_start(parsec);
        PARSEC_CHECK_ERROR(rc, "parsec_context_start");
        /***** Start of timing overhead of task generation ******/
        total_tasks = 1;
        nb = 1; /* size of each tile */
        nt = 1; /* total tiles */

        dtd_tp = parsec_dtd_taskpool_new();

        parsec_matrix_add2arena_rect(parsec_dtd_arenas[0],
                                     parsec_datatype_int32_t,
                                     nb, 1, nb);

        parsec_data_copy_t *gdata;
        parsec_data_t *data;
        int *real_data, key;

        dcA = create_and_distribute_data(rank, world, nb, nt);
        parsec_data_collection_set_key((parsec_data_collection_t *)dcA, "A");

        parsec_data_collection_t *A = (parsec_data_collection_t *)dcA;
        parsec_dtd_data_collection_init(A);

        if(rank == 0) {
            key = A->data_key(A, rank, 0);
            data = A->data_of_key(A, key);
            gdata = data->device_copies[0];
            real_data = PARSEC_DATA_COPY_GET_PTR((parsec_data_copy_t *) gdata);
            *real_data = rank;
            parsec_output( 0, "1: We pass data from rank 0 to 1 and flush it back\n");
        }

        rc = parsec_context_add_taskpool(parsec, dtd_tp);
        PARSEC_CHECK_ERROR(rc, "parsec_context_add_taskpool");

        int execute_in_rank = 1;
        parsec_dtd_taskpool_insert_task(dtd_tp, task_to_check_overhead_1,  0,  "task_for_timing_overhead",
                                        sizeof(int),      &execute_in_rank,     VALUE | AFFINITY,
                                        PASSED_BY_REF,    TILE_OF_KEY(A, 0),    INOUT,
                                        PARSEC_DTD_ARG_END);

        parsec_dtd_data_flush_all(dtd_tp, A);

        /* finishing all the tasks inserted, but not finishing the handle */
        rc = parsec_dtd_taskpool_wait( parsec, dtd_tp );
        PARSEC_CHECK_ERROR(rc, "parsec_dtd_taskpool_wait");

        parsec_taskpool_free(dtd_tp);

        rc = parsec_context_wait(parsec);
        PARSEC_CHECK_ERROR(rc, "parsec_context_wait");

        if(rank == 0) {
            key = A->data_key(A, rank, 0);
            data = A->data_of_key(A, key);
            gdata = data->device_copies[0];
            real_data = PARSEC_DATA_COPY_GET_PTR((parsec_data_copy_t *) gdata);
            assert(*real_data == 1);
            parsec_output( 0, "1: test PASSED\n");
        }
        parsec_dtd_data_collection_fini(A);
        free_data(dcA);


        /**** We send data from rank 0 to 1 and flush it back */
        /* Following the patter: rank:operation
         * RW:0
         * RW:1
         * RW:0
         * FLUSH
         */

        /* Registering the dtd_handle with PARSEC context */
        rc = parsec_context_start(parsec);
        PARSEC_CHECK_ERROR(rc, "parsec_context_start");
        /***** Start of timing overhead of task generation ******/
        dtd_tp = parsec_dtd_taskpool_new();

        dcA = create_and_distribute_data(rank, world, nb, nt);
        parsec_data_collection_set_key((parsec_data_collection_t *)dcA, "A");

        A = (parsec_data_collection_t *)dcA;
        parsec_dtd_data_collection_init(A);

        if(rank == 0) {
            key = A->data_key(A, rank, 0);
            data = A->data_of_key(A, key);
            gdata = data->device_copies[0];
            real_data = PARSEC_DATA_COPY_GET_PTR((parsec_data_copy_t *) gdata);
            *real_data = rank;
            parsec_output( 0, "2: We pass data from rank 0 to 1 and back to 0 and then try flushing it\n");
        }

        rc = parsec_context_add_taskpool(parsec, dtd_tp);
        PARSEC_CHECK_ERROR(rc, "parsec_context_add_taskpool");

        execute_in_rank = 0;
        parsec_dtd_taskpool_insert_task(dtd_tp, task_for_test_2_dist_mem,  0,  "task_for_timing_overhead",
                                        sizeof(int),      &execute_in_rank,     VALUE | AFFINITY,
                                        PASSED_BY_REF,    TILE_OF_KEY(A, 0),    INOUT,
                                        PARSEC_DTD_ARG_END);
        execute_in_rank = 1;
        parsec_dtd_taskpool_insert_task(dtd_tp, task_for_test_2_dist_mem,  0,  "task_for_timing_overhead",
                                        sizeof(int),      &execute_in_rank,     VALUE | AFFINITY,
                                        PASSED_BY_REF,    TILE_OF_KEY(A, 0),    INOUT,
                                        PARSEC_DTD_ARG_END);
        execute_in_rank = 0;
        parsec_dtd_taskpool_insert_task(dtd_tp, task_for_test_2_dist_mem,  0,  "task_for_timing_overhead",
                                        sizeof(int),      &execute_in_rank,     VALUE | AFFINITY,
                                        PASSED_BY_REF,    TILE_OF_KEY(A, 0),    INOUT,
                                        PARSEC_DTD_ARG_END);

        parsec_dtd_data_flush_all(dtd_tp, A);

        /* finishing all the tasks inserted, but not finishing the handle */
        rc = parsec_dtd_taskpool_wait( parsec, dtd_tp );
        PARSEC_CHECK_ERROR(rc, "parsec_dtd_taskpool_wait");

        parsec_taskpool_free(dtd_tp);

        rc = parsec_context_wait(parsec);
        PARSEC_CHECK_ERROR(rc, "parsec_context_wait");

        if(rank == 0) {
            key = A->data_key(A, rank, 0);
            data = A->data_of_key(A, key);
            gdata = data->device_copies[0];
            real_data = PARSEC_DATA_COPY_GET_PTR((parsec_data_copy_t *) gdata);
            assert(*real_data == 21);
            parsec_output( 0, "2: test PASSED\n");
        }
        parsec_dtd_data_collection_fini(A);
        free_data(dcA);
    } else if (world == 3) {
        /* We send data from rank 0 to 2 and flush it back
         * rank 1 does nothing.
         */

        /* Registering the dtd_handle with PARSEC context */
        rc = parsec_context_start(parsec);
        PARSEC_CHECK_ERROR(rc, "parsec_context_start");
        /***** Start of timing overhead of task generation ******/
        total_tasks = 1;
        nb = 1; /* size of each tile */
        nt = 1; /* total tiles */

        dtd_tp = parsec_dtd_taskpool_new();

        parsec_matrix_add2arena_rect(parsec_dtd_arenas[0],
                                     parsec_datatype_int32_t,
                                     nb, 1, nb);
        parsec_data_copy_t *gdata;
        parsec_data_t *data;
        int *real_data, key;

        dcA = create_and_distribute_data(rank, world, nb, nt);
        parsec_data_collection_set_key((parsec_data_collection_t *)dcA, "A");

        parsec_data_collection_t *A = (parsec_data_collection_t *)dcA;
        parsec_dtd_data_collection_init(A);

        if(rank == 0) {
            key = A->data_key(A, rank, 0);
            data = A->data_of_key(A, key);
            gdata = data->device_copies[0];
            real_data = PARSEC_DATA_COPY_GET_PTR((parsec_data_copy_t *) gdata);
            *real_data = rank;
            parsec_output( 0, "3: We pass data from rank 0 to 2 and flush it, rank 1 has no involvement\n");
        }


        rc = parsec_context_add_taskpool(parsec, dtd_tp);
        PARSEC_CHECK_ERROR(rc, "parsec_context_add_taskpool");

        int execute_in_rank = 2;
        parsec_dtd_taskpool_insert_task(dtd_tp, task_to_check_overhead_1,  0,  "task_for_timing_overhead",
                                        sizeof(int),      &execute_in_rank,     VALUE | AFFINITY,
                                        PASSED_BY_REF,    TILE_OF_KEY(A, 0),    INOUT,
                                        PARSEC_DTD_ARG_END);

        parsec_dtd_data_flush_all(dtd_tp, A);

        /* finishing all the tasks inserted, but not finishing the handle */
        rc = parsec_dtd_taskpool_wait( parsec, dtd_tp );
        PARSEC_CHECK_ERROR(rc, "parsec_dtd_taskpool_wait");

        parsec_taskpool_free(dtd_tp);

        rc = parsec_context_wait(parsec);
        PARSEC_CHECK_ERROR(rc, "parsec_context_wait");

        if(rank == 0) {
            key = A->data_key(A, rank, 0);
            data = A->data_of_key(A, key);
            gdata = data->device_copies[0];
            real_data = PARSEC_DATA_COPY_GET_PTR((parsec_data_copy_t *) gdata);
            assert(*real_data == 1);
            parsec_output( 0, "3: test PASSED\n");
        }
        parsec_dtd_data_collection_fini(A);
        free_data(dcA);
    }

    parsec_fini(&parsec);

#ifdef PARSEC_HAVE_MPI
    MPI_Finalize();
#endif

    return 0;
}
