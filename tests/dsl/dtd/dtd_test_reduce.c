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
task_rank_0( parsec_execution_stream_t    *es,
             parsec_task_t *this_task )
{
    (void)es;
    int *data;

    parsec_dtd_unpack_args(this_task, &data);

    if(this_task->taskpool->context->my_rank == 5)
        sleep(1);

    return PARSEC_HOOK_RETURN_DONE;
}

int
task_rank_1( parsec_execution_stream_t    *es,
             parsec_task_t *this_task )
{
    (void)es;
    int *data;
    int *second_data;

    parsec_dtd_unpack_args(this_task, &data, &second_data);

    *second_data += *data;
    printf( "My rank: %d, diff: %d\n", this_task->taskpool->context->my_rank, *data );

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

    nb = 1; /* tile_size */
    nt = world; /* total no. of tiles */

    if(argv[1] != NULL){
        cores = atoi(argv[1]);
    }

    parsec = parsec_init( cores, &argc, &argv );

    parsec_taskpool_t *dtd_tp = parsec_dtd_taskpool_new(  );

    adt = parsec_dtd_create_arena_datatype(parsec, &TILE_FULL);
    parsec_add2arena_rect( adt,
                                  parsec_datatype_int32_t,
                                  nb, 1, nb );

    /* Correctness checking */
    dcA = create_and_distribute_data(rank, world, nb, nt);
    parsec_data_collection_set_key((parsec_data_collection_t *)dcA, "A");

    parsec_data_collection_t *A = (parsec_data_collection_t *)dcA;
    parsec_dtd_data_collection_init(A);

    parsec_data_copy_t *gdata;
    parsec_data_t *data;
    int *real_data, key;
    int root = 0, i;

    /* Registering the dtd_handle with PARSEC context */
    rc = parsec_context_add_taskpool( parsec, dtd_tp );
    PARSEC_CHECK_ERROR(rc, "parsec_context_add_taskpool");
    rc = parsec_context_start(parsec);
    PARSEC_CHECK_ERROR(rc, "parsec_context_start");

    // *********************
    if( rank == root) {
        printf("Root: %d\n\n", root );
    }

    key = A->data_key(A, rank, 0);
    data = A->data_of_key(A, key);
    gdata = data->device_copies[0];
    real_data = PARSEC_DATA_COPY_GET_PTR((parsec_data_copy_t *) gdata);
    *real_data = rank;

    for( i = 0; i < world; i ++ ) {
        if( root != i ) {
            parsec_dtd_insert_task(dtd_tp, task_rank_0, 0, PARSEC_DEV_CPU, "task_rank_0",
                                   PASSED_BY_REF, PARSEC_DTD_TILE_OF_KEY(A, i), PARSEC_INOUT | TILE_FULL | PARSEC_AFFINITY,
                                   PARSEC_DTD_ARG_END);

            parsec_dtd_insert_task(dtd_tp, task_rank_1, 0, PARSEC_DEV_CPU, "task_rank_0",
                                   PASSED_BY_REF, PARSEC_DTD_TILE_OF_KEY(A, i),    PARSEC_INOUT | TILE_FULL,
                                   PASSED_BY_REF, PARSEC_DTD_TILE_OF_KEY(A, root), PARSEC_INOUT | TILE_FULL | PARSEC_AFFINITY,
                                   PARSEC_DTD_ARG_END);
        }
    }
    //******************
    parsec_dtd_data_flush_all( dtd_tp, A );

    rc = parsec_dtd_taskpool_wait( dtd_tp );
    PARSEC_CHECK_ERROR(rc, "parsec_dtd_taskpool_wait");
    rc = parsec_context_wait(parsec);
    PARSEC_CHECK_ERROR(rc, "parsec_context_wait");

    parsec_taskpool_free( dtd_tp );

    parsec_del2arena(adt);
    PARSEC_OBJ_RELEASE(adt->arena);
    parsec_dtd_destroy_arena_datatype(parsec, TILE_FULL);
    parsec_dtd_data_collection_fini( A );
    free_data(dcA);

    parsec_fini(&parsec);

#ifdef PARSEC_HAVE_MPI
    MPI_Finalize();
#endif

    return 0;
}
