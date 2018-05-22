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

enum regions {
               TILE_FULL,
             };

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

    nb = 1; /* tile_size */
    nt = world; /* total no. of tiles */

    if(argv[1] != NULL){
        cores = atoi(argv[1]);
    }

    parsec = parsec_init( cores, &argc, &argv );

    parsec_taskpool_t *dtd_tp = parsec_dtd_taskpool_new();

    parsec_matrix_add2arena_rect(parsec_dtd_arenas[TILE_FULL],
                                 parsec_datatype_int32_t,
                                 nb, 1, nb);

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

    for( i = 0; i < world - 1; i++ ) {
        parsec_dtd_taskpool_insert_task( dtd_tp, task_rank_0,    0,  "task_rank_0",
                            PASSED_BY_REF,     TILE_OF_KEY(A, A->data_key(A, i, 0)),   INOUT | TILE_FULL | AFFINITY,
                            PARSEC_DTD_ARG_END );
        parsec_dtd_taskpool_insert_task( dtd_tp, task_rank_1,    0,  "task_rank_1",
                            PASSED_BY_REF,     TILE_OF_KEY(A, A->data_key(A, i, 0)),   INOUT | TILE_FULL,
                            PASSED_BY_REF,     TILE_OF_KEY(A, A->data_key(A, i+1, 0)), INOUT | TILE_FULL | AFFINITY,
                            PARSEC_DTD_ARG_END );
    }

    rc = parsec_dtd_taskpool_wait( parsec, dtd_tp );
    PARSEC_CHECK_ERROR(rc, "parsec_dtd_taskpool_wait");

    rc = parsec_dtd_taskpool_wait( parsec, dtd_tp );
    PARSEC_CHECK_ERROR(rc, "parsec_dtd_taskpool_wait");

    for( i = 0; i < world - 1; i++ ) {
        parsec_dtd_taskpool_insert_task( dtd_tp, task_rank_0,    0,  "task_rank_0",
                            PASSED_BY_REF,     TILE_OF_KEY(A, A->data_key(A, i, 0)),   INOUT | TILE_FULL | AFFINITY,
                            PARSEC_DTD_ARG_END );
        parsec_dtd_taskpool_insert_task( dtd_tp, task_rank_1,    0,  "task_rank_1",
                            PASSED_BY_REF,     TILE_OF_KEY(A, A->data_key(A, i, 0)),   INOUT | TILE_FULL,
                            PASSED_BY_REF,     TILE_OF_KEY(A, A->data_key(A, i+1, 0)), INOUT | TILE_FULL | AFFINITY,
                            PARSEC_DTD_ARG_END );
    }

    parsec_dtd_data_flush_all( dtd_tp, A );
    rc = parsec_dtd_taskpool_wait( parsec, dtd_tp );
    PARSEC_CHECK_ERROR(rc, "parsec_dtd_taskpool_wait");
    parsec_taskpool_free( dtd_tp );

    parsec_context_wait(parsec);

    parsec_arena_destruct(parsec_dtd_arenas[0]);
    parsec_dtd_data_collection_fini( A );
    free_data(dcA);

    parsec_fini(&parsec);

#ifdef PARSEC_HAVE_MPI
    MPI_Finalize();
#endif

    return 0;
}
