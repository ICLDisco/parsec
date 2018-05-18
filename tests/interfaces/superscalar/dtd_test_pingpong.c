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

enum regions {
               TILE_FULL,
             };

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

    parsec_taskpool_t *dtd_tp = parsec_dtd_taskpool_new(  );

    parsec_matrix_add2arena_rect(parsec_dtd_arenas[TILE_FULL],
                                 parsec_datatype_int32_t,
                                 nb, 1, nb);

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

    parsec_dtd_taskpool_insert_task(dtd_tp, task_rank_0,    0,  "task_rank_0",
                                    PASSED_BY_REF,    TILE_OF_KEY(A, 0), INOUT | TILE_FULL | AFFINITY,
                                    PARSEC_DTD_ARG_END);
    parsec_dtd_taskpool_insert_task(dtd_tp, task_rank_1,    0,  "task_rank_1",
                                    PASSED_BY_REF,    TILE_OF_KEY(A, 0), INOUT | TILE_FULL,
                                    PASSED_BY_REF,    TILE_OF_KEY(A, 1), INOUT | TILE_FULL | AFFINITY,
                                    PARSEC_DTD_ARG_END);

    parsec_dtd_data_flush_all( dtd_tp, A );

    rc = parsec_dtd_taskpool_wait( parsec, dtd_tp );
    PARSEC_CHECK_ERROR(rc, "parsec_dtd_taskpool_wait");
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

    parsec_arena_destruct(parsec_dtd_arenas[0]);
    parsec_dtd_data_collection_fini( A );
    free_data(dcA);

    if( 0 == rank ) {
        parsec_output( 0, "\nPingpong is behaving correctly.\n" );
    }

    parsec_taskpool_free( dtd_tp );

    /* End of correctness checking */


    /* Start of Pingpong timing */
    if( 0 == rank ) {
        parsec_output( 0, "\nChecking time of pingpong. We send data from rank 0 to rank 1 "
                       "And vice versa.\nWe perform this pingpong for 1000 times and measure the time. "
                       "We report the time for different size of data.\n\n" );
    }

    int repeat_pingpong = 1000;
    int sizes_of_data = 4, i, j;
    int sizes[4] = {100, 1000, 10000, 100000};


    for( i = 0; i < sizes_of_data; i++ ) {
        dtd_tp = parsec_dtd_taskpool_new(  );

        rc = parsec_context_add_taskpool( parsec, dtd_tp );
        PARSEC_CHECK_ERROR(rc, "parsec_context_add_taskpool");
        rc = parsec_context_start(parsec);
        PARSEC_CHECK_ERROR(rc, "parsec_context_start");

        nb = sizes[i];
        nt = 2;

        parsec_matrix_add2arena_rect(parsec_dtd_arenas[TILE_FULL],
                                     parsec_datatype_int32_t,
                                     nb, 1, nb);

        dcA = create_and_distribute_data(rank, world, nb, nt);
        parsec_data_collection_set_key((parsec_data_collection_t *)dcA, "A");

        parsec_data_collection_t *A = (parsec_data_collection_t *)dcA;
        parsec_dtd_data_collection_init(A);

        SYNC_TIME_START();

        for( j = 0; j < repeat_pingpong; j++ ) {
            parsec_dtd_taskpool_insert_task(dtd_tp, task_rank_0,    0,  "task_for_timing_0",
                                            PASSED_BY_REF,    TILE_OF_KEY(A, 0), INOUT | TILE_FULL | AFFINITY,
                                            PARSEC_DTD_ARG_END);
            parsec_dtd_taskpool_insert_task(dtd_tp, task_rank_1,    0,  "task_for_timing_1",
                                            PASSED_BY_REF,    TILE_OF_KEY(A, 0), INOUT | TILE_FULL,
                                            PASSED_BY_REF,    TILE_OF_KEY(A, 1), INOUT | TILE_FULL | AFFINITY,
                                            PARSEC_DTD_ARG_END);
        }
        parsec_dtd_data_flush_all( dtd_tp, A );
        /* finishing all the tasks inserted, but not finishing the handle */
        rc = parsec_dtd_taskpool_wait( parsec, dtd_tp );
        PARSEC_CHECK_ERROR(rc, "parsec_dtd_taskpool_wait");

        rc = parsec_context_wait(parsec);
        PARSEC_CHECK_ERROR(rc, "parsec_context_wait");
        SYNC_TIME_PRINT(rank, ("\tSize of message : %ld bytes\tTime for each pingpong : %12.5f\n", sizes[i]*sizeof(int), sync_time_elapsed/repeat_pingpong));

        parsec_arena_destruct(parsec_dtd_arenas[0]);
        parsec_dtd_data_collection_fini( A );
        free_data(dcA);
    }

    parsec_fini(&parsec);

#ifdef PARSEC_HAVE_MPI
    MPI_Finalize();
#endif

    return 0;
}
