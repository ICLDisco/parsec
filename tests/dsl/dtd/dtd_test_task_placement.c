/* parsec things */
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

/* IDs for the Arena Datatypes */
static int TILE_FULL;

int
task_task_placement(parsec_execution_stream_t *es,
                    parsec_task_t *this_task)
{
    (void)es;
    int intended_rank;

    parsec_dtd_unpack_args(this_task, &intended_rank);

    if(this_task->taskpool->context->nb_nodes <= intended_rank) {
        assert(this_task->taskpool->context->my_rank == 0);
        printf("Task placed in: %d and it is being executed in: %d\n", intended_rank, this_task->taskpool->context->my_rank);
    } else {
        assert(this_task->taskpool->context->my_rank == intended_rank);
        printf("Task placed in: %d and it is being executed in: %d\n", intended_rank, this_task->taskpool->context->my_rank);
    }

    return PARSEC_HOOK_RETURN_DONE;
}

int
task_precedence(parsec_execution_stream_t *es,
               parsec_task_t *this_task)
{
    (void)es;
    int intended_rank_1, intended_rank_2;
    int *data;

    parsec_dtd_unpack_args(this_task, &intended_rank_1, &data, &intended_rank_2);

    assert(this_task->taskpool->context->my_rank == 1);
    assert(intended_rank_1 == 1);
    printf("Intended rank was: %d and executed in: %d\n", intended_rank_1, this_task->taskpool->context->my_rank);

    return PARSEC_HOOK_RETURN_DONE;
}

int
task_moving_data(parsec_execution_stream_t *es,
                 parsec_task_t *this_task)
{
    (void)es;
    int intended_rank;
    int *data;

    parsec_dtd_unpack_args(this_task, &intended_rank, &data);

    assert(this_task->taskpool->context->my_rank == intended_rank);
    assert(*data == 20);
    printf("Task getting executed: %d data is: %d\n", intended_rank, *data);

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
    nt = world; /* total no. of tiles */

    parsec = parsec_init(cores, &argc, &argv);

    parsec_taskpool_t *dtd_tp = parsec_dtd_taskpool_new();

    adt = PARSEC_OBJ_NEW(parsec_arena_datatype_t);
    parsec_matrix_adt_construct_rect(adt,
            parsec_datatype_int32_t, nb, 1, nb);
    parsec_dtd_attach_arena_datatype(parsec, adt, &TILE_FULL);

    /* Correctness checking */
    dcA = create_and_distribute_data(rank, world, nb, nt);
    parsec_data_collection_set_key((parsec_data_collection_t *)dcA, "A");
    parsec_data_collection_t *A = (parsec_data_collection_t *)dcA;
    parsec_dtd_data_collection_init(A);

    int intended_rank = 0, i;

    if(rank == 0) {
        /* Initializing data */
        parsec_data_copy_t *gdata;
        parsec_data_t *data;
        int *real_data;

        data = A->data_of_key(A, 0);
        gdata = data->device_copies[0];
        real_data = PARSEC_DATA_COPY_GET_PTR((parsec_data_copy_t *) gdata);
        /* Setting the data as 20 */
        *real_data = 20;
    }

    /* Registering the dtd_handle with PARSEC context */
    rc = parsec_context_add_taskpool( parsec, dtd_tp );
    PARSEC_CHECK_ERROR(rc, "parsec_context_add_taskpool");
    rc = parsec_context_start(parsec);
    PARSEC_CHECK_ERROR(rc, "parsec_context_start");

    for(i = 0; i < 1000; i++) {
        /* Testing AFFINITY flag with value */
        intended_rank = 1;
        parsec_dtd_insert_task(dtd_tp, task_task_placement, 0, PARSEC_DEV_CPU, "task_task_placement",
                               sizeof(int), &intended_rank,              PARSEC_VALUE | PARSEC_AFFINITY,
                               PARSEC_DTD_ARG_END);

        intended_rank = 2;
        parsec_dtd_insert_task(dtd_tp, task_task_placement, 0, PARSEC_DEV_CPU, "task_task_placement",
                               sizeof(int), &intended_rank,              PARSEC_VALUE | PARSEC_AFFINITY,
                               PARSEC_DTD_ARG_END);


        intended_rank = 1;
        printf("Using affinity with data residing in rank: %d\n", A->rank_of_key(A, 0));
        parsec_dtd_insert_task(dtd_tp, task_precedence, 0, PARSEC_DEV_CPU, "task_precedence",
                               sizeof(int), &intended_rank,               PARSEC_VALUE | PARSEC_AFFINITY,
                               PASSED_BY_REF, PARSEC_DTD_TILE_OF_KEY(A, 0), PARSEC_INOUT | TILE_FULL | PARSEC_AFFINITY,
                               sizeof(int), &intended_rank,               PARSEC_VALUE | PARSEC_AFFINITY,
                               PARSEC_DTD_ARG_END);

        /* Data reside in rank 0 and we set the data to 20,
         * and ask the task to be executed in rank 1. Correct
         * behavior would be to find "20" as data in rank 1.
         */
        intended_rank = 1;
        parsec_dtd_insert_task(dtd_tp, task_moving_data, 0, PARSEC_DEV_CPU, "task_moving",
                               sizeof(int), &intended_rank,    PARSEC_VALUE | PARSEC_AFFINITY,
                               PASSED_BY_REF, PARSEC_DTD_TILE_OF_KEY(A, 0), PARSEC_INOUT | TILE_FULL,
                               PARSEC_DTD_ARG_END);
    }

    parsec_dtd_data_flush_all( dtd_tp, A );

    rc = parsec_taskpool_wait( dtd_tp );
    PARSEC_CHECK_ERROR(rc, "parsec_taskpool_wait");
    rc = parsec_context_wait(parsec);
    PARSEC_CHECK_ERROR(rc, "parsec_context_wait");

    parsec_taskpool_free( dtd_tp );

    parsec_dtd_free_arena_datatype(parsec, TILE_FULL);
    parsec_dtd_data_collection_fini( A );
    free_data(dcA);

    parsec_fini(&parsec);

#ifdef PARSEC_HAVE_MPI
    MPI_Finalize();
#endif

    return 0;
}
